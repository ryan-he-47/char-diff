"""Online AE-quality verify + resplit fallback (2026-09-29 用户拍板:在线做,
不碰缓存——编码器/阈值迭代不需要重建缓存).

Gate (双判据,用户拍板): a cached unit is REPLACED when the frozen AE either
  * fails exact-match reconstruction (EM fail), or
  * decodes correctly but with mean byte-entropy > ent_thresh (fragile —
    "不能产生质量好的嵌入向量,应当重切").
Cascade: source unit -> raw BPE pieces (<= l_max) -> re-verify -> failing
pieces degrade to char-aware <= floor_bytes (default 3) chunks ("2~3 字节短片
让模型慢慢拼"; char-aware so UTF-8 chars are never split — the single-BYTE
floor measured 108/191 failures on bare continuation bytes). Floor chunks are
accepted unconditionally but still verified once for the residual counter.

Cost model: verification is ONE parallel FiLM forward (the decoder is not
autoregressive) and results are memoized per distinct unit per process
(measured occurrence repetition ~133x on the full shard -> steady-state GPU
cost ~0). Full-shard measurement (docs/39 #23): flag 0.115% @TH=0.2, unit
inflation +0.53%, threshold-insensitive in [0.1, 0.5].

Determinism: everything is a pure function of (AE ckpt, tokenizer, thresholds)
-> pin the AE ckpt per run; bpb comparability unaffected (byte-normalized).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .models.autoencoder import EOS_ID, ByteStringAE


def _char_chunks(b: bytes, cap: int) -> list[bytes]:
    """Char-aware <=cap-byte chunks (surrogateescape round-trips dirty bytes)."""
    s = b.decode("utf-8", errors="surrogateescape")
    out: list[bytes] = []
    cur = b""
    for ch in s:
        cb = ch.encode("utf-8", errors="surrogateescape")
        if cur and len(cur) + len(cb) > cap:
            out.append(cur)
            cur = b""
        cur += cb
    if cur:
        out.append(cur)
    return out


class VerifyResplitter:
    def __init__(
        self,
        ae: ByteStringAE,
        tok_dir: str,
        ent_thresh: float = 0.2,
        floor_bytes: int = 3,
        device: str = "cuda",
        log_every: int = 200,
    ):
        self.ae = ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.l_max = int(ae.l_max)
        self.ent_thresh = float(ent_thresh)
        self.floor_bytes = int(floor_bytes)
        self.device = device
        self.log_every = int(log_every)
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        self._cache: dict[bytes, tuple[bytes, ...]] = {}
        # stats (cumulative): units in, flagged, resplit pieces, floored pieces,
        # floor pieces still failing (accepted as-is, per user: 微量噪音先不管)
        self.n_in = 0
        self.n_flag = 0
        self.n_piece = 0
        self.n_floor = 0
        self.n_floor_fail = 0
        self.n_out = 0
        self.n_verify_rows = 0
        self._calls = 0

    # ---- AE quality probe: one parallel forward per chunk -----------------
    @torch.no_grad()
    def _verify(self, units: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
        """-> (em_ok (N,), mean byte-entropy bits (N,)). fp32 (calibration-
        matching: all acceptance numbers were measured in fp32)."""
        N = len(units)
        ems = np.empty(N, dtype=bool)
        ents = np.empty(N, dtype=np.float32)
        for i in range(0, N, 8192):
            chunk = units[i : i + 8192]
            B = len(chunk)
            arr = np.zeros((B, self.l_max), dtype=np.uint8)
            lens = np.zeros(B, dtype=np.int64)
            for j, u in enumerate(chunk):
                n = min(len(u), self.l_max)  # oversized probes clamp ...
                arr[j, :n] = np.frombuffer(u[:n], dtype=np.uint8)
                lens[j] = n
            ids = torch.from_numpy(arr).long().to(self.device)
            lens_t = torch.from_numpy(lens).to(self.device)
            pm = torch.arange(self.l_max, device=self.device)[None, :] >= lens_t[:, None]
            logits = self.ae(ids, pm).float()  # (B, K=l_max+1, 257)
            logp = torch.log_softmax(logits, -1)
            h = -(logp.exp() * logp).sum(-1) / np.log(2.0)  # (B, K) bits
            K = self.l_max + 1
            d = torch.arange(K, device=self.device)[None, :]  # query Δ = d+1
            valid = d <= lens_t[:, None]  # bytes at d<len, EOS slot at d=len
            tgt = torch.where(
                d == lens_t[:, None],
                torch.full_like(ids[:, :1], EOS_ID).expand(B, K),
                torch.cat([ids, torch.zeros(B, 1, dtype=torch.long,
                                            device=self.device)], dim=1),
            )
            pred = logits.argmax(-1)
            em = ((pred == tgt) | ~valid).all(dim=1).cpu().numpy()
            byte_mask = (d < lens_t[:, None]).float()
            ent = ((h * byte_mask).sum(1) / byte_mask.sum(1).clamp(min=1))
            ems[i : i + B] = em
            ents[i : i + B] = ent.cpu().numpy()
            self.n_verify_rows += B
            for j, u in enumerate(chunk):
                if len(u) > self.l_max:  # ... then force-fail: cannot encode
                    ems[i + j] = False   # -> always resplit (pieces <= l_max)
                    ents[i + j] = float("inf")
        return ems, ents

    def _bpe_split(self, b: bytes) -> list[bytes]:
        """Raw Qwen BPE pieces, hard-capped at l_max (mirrors segment_v2)."""
        try:
            s = b.decode("utf-8", errors="surrogateescape")
            enc = self._tok.encode(s, add_special_tokens=False)
        except Exception:
            return _char_chunks(b, self.floor_bytes)
        ords = np.fromiter(map(ord, s), dtype=np.int64, count=len(s))
        widths = np.where(
            ords < 0x80, 1,
            np.where((ords >= 0xDC80) & (ords <= 0xDCFF), 1,
                     np.where(ords < 0x800, 2,
                              np.where(ords < 0x10000, 3, 4))))
        cum = np.concatenate([[0], np.cumsum(widths)])
        cuts = sorted({int(cum[st]) for st, _ in enc.offsets
                       if 0 < int(cum[st]) < len(b)})
        out, pos = [], 0
        for g in cuts + [len(b)]:
            piece = b[pos:g]
            pos = g
            for i in range(0, len(piece), self.l_max):
                if piece[i : i + self.l_max]:
                    out.append(piece[i : i + self.l_max])
        return out

    # ---- main entry --------------------------------------------------------
    def _ensure(self, misses: list[bytes]) -> None:
        """Verify + cascade for cache misses, ONE GPU roundtrip per call
        (2026-09-29 slowdown fix: per-window roundtrips cost ~6ms each;
        batching across the whole training batch cuts 36 -> 1)."""
        if not misses:
            return
        em_ok, ent = self._verify(misses)
        fails = [u for u, ok, e in zip(misses, em_ok, ent)
                 if (not ok) or e > self.ent_thresh]
        self.n_flag += len(fails)
        for u, ok, e in zip(misses, em_ok, ent):
            if ok and e <= self.ent_thresh:
                self._cache[u] = (u,)
        if not fails:
            return
        pieces = list({p for u in fails for p in self._bpe_split(u)})
        pieces = [p for p in pieces if p not in self._cache]
        if pieces:
            p_ok, p_ent = self._verify(pieces)
            self.n_piece += len(pieces)
            floor: list[bytes] = []
            for p, ok, e in zip(pieces, p_ok, p_ent):
                if ok and e <= self.ent_thresh:
                    self._cache[p] = (p,)
                else:
                    floor.extend(_char_chunks(p, self.floor_bytes))
            floor = [f for f in dict.fromkeys(floor) if f not in self._cache]
            if floor:
                f_ok, f_ent = self._verify(floor)
                self.n_floor += len(floor)
                self.n_floor_fail += int(sum(
                    1 for ok, e in zip(f_ok, f_ent)
                    if (not ok) or e > self.ent_thresh))
                for f in floor:
                    self._cache[f] = (f,)  # accepted unconditionally
        for u in fails:
            finals: list[bytes] = []
            for p in self._bpe_split(u):
                if p in self._cache and self._cache[p] == (p,):
                    finals.append(p)
                else:
                    finals.extend(_char_chunks(p, self.floor_bytes))
            self._cache[u] = tuple(finals)

    def process_batch(
        self, windows: list[list[bytes]], doc_starts: list[list[bool]] | None = None
    ):
        """Verify + resplit a whole batch of windows. Byte-conserving per
        window; counts only grow (splits). With doc_starts given (2026-10-03
        边界掩码), also returns propagated flags: the first piece of a resplit
        doc-start unit keeps the flag, the rest are False."""
        self._calls += len(windows)
        self.n_in += sum(len(w) for w in windows)
        misses = [u for w in windows for u in dict.fromkeys(w)
                  if u not in self._cache]
        self._ensure(list(dict.fromkeys(misses)))
        out: list[list[bytes]] = []
        for w in windows:
            o: list[bytes] = []
            for u in w:
                o.extend(self._cache[u])
            out.append(o)
            self.n_out += len(o)
        if self.log_every and self._calls // self.log_every != (
                self._calls - len(windows)) // self.log_every:
            print(f"[verify] calls {self._calls} | units {self.n_in} "
                  f"flag {self.n_flag} ({self.n_flag / max(self.n_in, 1):.4%}) "
                  f"| in->out x{self.n_out / max(self.n_in, 1):.4f} "
                  f"| cache {len(self._cache)} | verify rows {self.n_verify_rows} "
                  f"| floor {self.n_floor} (fail {self.n_floor_fail})", flush=True)
        if doc_starts is None:
            return out
        out_st: list[list[bool]] = []
        for w, st in zip(windows, doc_starts):
            o_st: list[bool] = []
            for u, f in zip(w, st):
                o_st.extend([bool(f)] + [False] * (len(self._cache[u]) - 1))
            out_st.append(o_st)
        return out, out_st

    def process(self, units: list[bytes]) -> list[bytes]:
        """Single-window wrapper (kept for tests/diagnostics)."""
        return self.process_batch([units])[0]


def fit_window(units: list[bytes], S: int) -> list[bytes]:
    """Trim a verified window to exactly S units (counts only grow under
    resplit; random-offset sampling makes the dropped tail bytes a uniform
    tiny loss — 2026-09-29 拍板)."""
    return units[:S]
