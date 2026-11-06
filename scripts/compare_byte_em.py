"""Shared text-history / byte-span EM for latent diffusion and a BPE AR model."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.flow import generate
from char_diff.gmm import mixture_mean, ode_step
from char_diff.model import Denoiser, BOS, CLEAN, NOISY, PAD
from char_diff.utils import ROOT, sha256, write_json

BASELINE = ROOT / 'assets/public_baselines/tinystories_custom20m'
MODELS = {'x0': 'runs/continuation_review_20261101/uniform/last.pt',
          'gmm_early': 'runs/gmm_review_20261102/early/last.pt',
          'gmm_late': 'runs/gmm_review_20261102/late/last.pt'}


def vocabulary_bytes(tokenizer):
    # GPT-2's reversible byte alphabet, kept here because recent Transformers
    # tokenizers no longer export the old bytes_to_unicode helper.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [b for b in range(256) if b not in visible]
    inverse = {chr(b): b for b in visible}
    inverse.update({chr(256 + i): b for i, b in enumerate(missing)})
    return [b'' if i in tokenizer.all_special_ids else bytes(inverse[c] for c in tokenizer.convert_ids_to_tokens(i))
            for i in range(len(tokenizer))]


def prepare_cases(cache, tokenizer, windows=64, context_tokens=192):
    offsets = np.fromfile(cache.root / 'unit_offsets.u64', dtype='<u8')
    blob = (cache.root / 'units.bin').read_bytes()
    raw = lambda i: blob[int(offsets[i]):int(offsets[i + 1])]
    selected = np.random.default_rng(20261101).choice(len(cache), min(windows, len(cache)), replace=False).tolist()
    cases = []
    for window in selected:
        parts = cache.segments[int(cache.windows[window]):int(cache.windows[window + 1])]
        for doc, start, end, prior in parts:
            doc, start, end, prior = map(int, (doc, start, end, prior))
            ids = cache.ids[int(cache.offsets[doc]):int(cache.offsets[doc + 1])]
            for anchor in range(start, end, cache.block):
                def encode_from(begin):
                    text = b''.join(raw(int(i)) for i in ids[begin:anchor])
                    return text, tokenizer.encode(text.decode('utf-8'), add_special_tokens=False)
                # Truncate only at AE-unit boundaries; both families receive
                # exactly these retained raw bytes, without target information.
                left, right = prior, anchor
                while left < right:
                    mid = (left + right) // 2
                    if len(encode_from(mid)[1]) > context_tokens: left = mid + 1
                    else: right = mid
                history, tokens = encode_from(left)
                assert len(tokens) <= context_tokens
                length = min(cache.block, end - anchor)
                reference = b''.join(raw(int(i)) for i in ids[anchor:anchor + length])
                cases.append({'id': len(cases), 'window': window, 'doc': doc, 'anchor': anchor,
                              'history_start': left, 'history_ids': [int(i) for i in ids[left:anchor]],
                              'history_hex': history.hex(), 'input_ids': tokens,
                              'block_length': length, 'target_hex': raw(int(ids[anchor])).hex(),
                              'reference_hex': reference[:32].hex()})
    return selected, cases


def byte_score(target, reference, predicted, native=None):
    length = len(target)
    exact = predicted[:length] == target and len(predicted) >= length
    return {'target_bytes': length, 'exact': int(exact), 'exact_bytes': length * int(exact),
            'em4': int(predicted[:4] == reference[:4] and len(predicted) >= 4) if len(reference) >= 4 else None,
            'em8': int(predicted[:8] == reference[:8] and len(predicted) >= 8) if len(reference) >= 8 else None,
            'native_unit_exact': int(native == target) if native is not None else None}


def aggregate(rows):
    def count(values):
        n = len(values); total_bytes = sum(v['target_bytes'] for v in values)
        result = {'cases': n, 'target_bytes': total_bytes, 'exact': sum(v['exact'] for v in values),
                  'exact_bytes': sum(v['exact_bytes'] for v in values)}
        result['span_em'] = result['exact'] / n if n else None
        result['byte_weighted_em'] = result['exact_bytes'] / total_bytes if total_bytes else None
        for key in ('em4', 'em8', 'native_unit_exact'):
            eligible = [v[key] for v in values if v[key] is not None]
            result[key] = sum(eligible) / len(eligible) if eligible else None
            result[key + '_cases'] = len(eligible)
        return result
    result = count(rows)
    result['by_length'] = {f'{lo}-{hi}': count([r for r in rows if lo <= r['target_bytes'] <= hi])
                           for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32))}
    result['by_seed'] = {str(s): count([r for r in rows if r['seed'] == s]) for s in sorted({r['seed'] for r in rows})}
    return result


def diffusion_batch(cache, cases, device='cuda'):
    b = len(cases); nc = 1 + max(len(c['history_ids']) for c in cases); nq = cache.block
    x = torch.zeros(b, nc, 48, device=device)
    roles = torch.full((b, nc), PAD, dtype=torch.long, device=device)
    pos = torch.zeros_like(roles); lengths = torch.zeros_like(roles)
    qp = torch.zeros(b, nq, dtype=torch.long, device=device)
    qr = torch.full_like(qp, PAD); ql = torch.zeros_like(qp)
    cv = torch.zeros(b, nc, dtype=torch.bool, device=device)
    qv = torch.zeros(b, nq, dtype=torch.bool, device=device)
    for i, case in enumerate(cases):
        hist = case['history_ids']; n = len(hist) + 1; length = case['block_length']
        if hist: x[i, 1:n] = torch.as_tensor(cache.vectors[hist], device=device)
        roles[i, 0] = BOS; roles[i, 1:n] = CLEAN; pos[i, :n] = torch.arange(n, device=device)
        cv[i, :n] = True; qv[i, :length] = True
        qr[i, :length] = NOISY; ql[i, :length] = length
        qp[i, :length] = torch.arange(n, n + length, device=device)
    cc = (cv[:, :, None] & cv[:, None, :] & torch.ones(nc, nc, device=device, dtype=torch.bool).tril())
    cc |= (~cv)[:, :, None] & torch.eye(nc, device=device, dtype=torch.bool)[None]
    nn = qv[:, :, None] & qv[:, None, :]
    nn |= (~qv)[:, :, None] & torch.eye(nq, device=device, dtype=torch.bool)[None]
    allowed = torch.cat([qv[:, :, None] & cv[:, None, :], nn], -1)[:, None]
    return x, roles, pos, lengths, cc[:, None], qr, qp, ql, allowed


@torch.inference_mode()
def diffusion_predict(model, packed, z, sampler, steps=32):
    x, roles, pos, lengths, cc, qr, qp, ql, allowed = packed
    _, past = model(x, torch.zeros_like(pos, dtype=torch.float32), roles, pos, lengths, cc, return_cache=True)
    components = model.cfg.get('gmm_components', 0)
    for iteration in range(steps):
        t = 1 - iteration / steps; tn = 1 - (iteration + 1) / steps
        pred = model(z, torch.full_like(qp, t, dtype=torch.float32), qr, qp, ql, allowed, past=past)
        if sampler == 'gm_ode': z = ode_step(pred, z, t, tn, float(model.cfg['gmm_sigma']), 4)
        else: z = (tn / t) * z + (1 - tn / t) * (mixture_mean(pred) if components else pred.float())
    if not torch.isfinite(z).all(): raise FloatingPointError('Non-finite conditional generation')
    return z


def healing_mask(required, byte_vocab, device):
    # Prefix constraints use only already observed bytes. A candidate either
    # finishes them or consumes a proper prefix before the next decoding step.
    return torch.tensor([bool(v) and (v.startswith(required) or required.startswith(v)) for v in byte_vocab],
                        dtype=torch.bool, device=device)


@torch.inference_mode()
def ar_predict(model, tokenizer, byte_vocab, cases, seed, greedy=False):
    device = next(model.parameters()).device
    histories = []; required = []; healed = []
    for case in cases:
        ids = list(case['input_ids']); tail = b''
        if ids and byte_vocab[ids[-1]].isspace():
            tail = byte_vocab[ids.pop()]
        histories.append(ids or [tokenizer.bos_token_id]); required.append(tail); healed.append(tail.hex())
    b = len(cases); width = max(map(len, histories))
    ids = torch.full((b, width), tokenizer.eos_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, history in enumerate(histories):
        ids[i, -len(history):] = torch.tensor(history, device=device); attention[i, -len(history):] = 1
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    output = model(ids, attention_mask=attention, position_ids=positions, use_cache=True)
    past = output.past_key_values; logits = output.logits[:, -1].float()
    generated = [b'' for _ in cases]; done = [False] * b; ended = [False] * b
    generator = torch.Generator(device=device).manual_seed(seed)
    masks = {}
    for iteration in range(64):
        for i, suffix in enumerate(required):
            if suffix and not done[i]:
                if suffix not in masks: masks[suffix] = healing_mask(suffix, byte_vocab, device)
                logits[i].masked_fill_(~masks[suffix], -torch.inf)
        chosen = logits.argmax(-1) if greedy else torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
        for i, ident in enumerate(chosen.tolist()):
            if done[i]: continue
            if ident == tokenizer.eos_token_id:
                assert not required[i]
                ended[i] = done[i] = True
                continue
            value = byte_vocab[ident]
            if required[i]:
                suffix = required[i]
                if value.startswith(suffix): value = value[len(suffix):]; required[i] = b''
                else:
                    assert suffix.startswith(value) and value
                    required[i] = suffix[len(value):]; value = b''
            generated[i] += value
            done[i] = len(generated[i]) >= 32
        if all(done): break
        attention = torch.cat([attention, torch.ones(b, 1, dtype=torch.long, device=device)], -1)
        positions = attention.sum(-1, keepdim=True) - 1
        output = model(chosen[:, None], past_key_values=past, attention_mask=attention,
                       position_ids=positions, use_cache=True)
        past = output.past_key_values; logits = output.logits[:, -1].float()
    assert all(done), 'AR generation reached the context-reserved 64-step cap'
    return generated, ended, healed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--windows', type=int, default=64)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--out', default='runs/cross_model_byte_em_20261102')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--only', choices=['diffusion', 'ar'])
    args = parser.parse_args(); torch.set_num_threads(4)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(BASELINE, local_files_only=True)
    byte_vocab = vocabulary_bytes(tokenizer)
    folder = ROOT / args.out; folder.mkdir(parents=True, exist_ok=True)
    cache = LatentCache(ROOT / 'data/tinystories', args.split)
    if (folder / 'anchors.json').exists():
        data = json.loads((folder / 'anchors.json').read_text(encoding='utf-8'))
        selected, cases = data['windows'], data['cases']
        assert data['split'] == args.split and len(selected) == args.windows
    else:
        selected, cases = prepare_cases(cache, tokenizer, args.windows)
        write_json(folder / 'anchors.json', {'split': args.split, 'windows': selected, 'cases': cases})
    protocol = {'split': args.split, 'window_count': len(selected), 'anchor_count': len(cases), 'selection_seed': 20261101,
                'seeds': [42, 123], 'batch': args.batch, 'steps': 32, 'precision': 'fp32', 'weights': 'ema',
                'history': 'same raw bytes, AE-boundary suffix fitting <=192 baseline BPE tokens; true block history',
                'score': 'first L generated raw bytes equal reference AE unit; byte-weighted EM=sum(L*hit)/sum(L)',
                'ar': 'temperature1, no truncation; greedy separate; trailing whitespace token healing on known bytes',
                'ar_batch_seed': 'seed + first_anchor_index * 1009; batch size fixed by protocol',
                'generation_horizon': 'native diffusion block; AR produces >=32 new bytes or EOS, reference length is scoring-only',
                'byte_horizons': [4, 8], 'baseline_revision': 'de881b7d0b0ce3f861a7cc4c3c0d5bdfb1238347',
                'baseline_model_sha256': sha256(BASELINE / 'model.safetensors'), 'cache_fingerprint': cache.fingerprint}
    write_json(folder / 'protocol.json', protocol)
    if args.prepare_only:
        print(json.dumps({'anchors': len(cases), 'windows': len(selected)}), flush=True); return
    report = {'protocol': protocol, 'models': {}}; started = time.perf_counter()
    def save_group(label, rows, metadata):
        report['models'][label] = {**metadata, 'metrics': aggregate(rows)}
        write_json(folder / f'{label}-summary.json', report['models'][label])
        print(json.dumps({'model': label, 'metrics': {k: v for k, v in report['models'][label]['metrics'].items() if not isinstance(v, dict)}}), flush=True)
    if args.only != 'ar':
        codec = Codec(device='cuda'); mean = torch.tensor(cache.mean, device='cuda')
        for name, path in MODELS.items():
            state = torch.load(ROOT / path, map_location='cpu', weights_only=False)
            assert state['cache_fingerprint'] == cache.fingerprint and state['codec'] == codec.fingerprint
            model = Denoiser(state['config']['model']).cuda().eval(); model.load_state_dict(state['ema'])
            methods = ['euler', 'gm_ode'] if model.cfg.get('gmm_components') else ['euler']
            rows = {method: [] for method in methods}
            streams = {s: torch.Generator(device='cuda').manual_seed(s) for s in [42, 123]}
            handles = {method: (folder / f'{name}_{method}.jsonl').open('x', encoding='utf-8', buffering=1) for method in methods}
            for begin in range(0, len(cases), args.batch):
                group = cases[begin:begin + args.batch]; packed = diffusion_batch(cache, group)
                for seed in [42, 123]:
                    noise = torch.stack([torch.randn(16, 48, device='cuda', generator=streams[seed]) for _ in group])
                    for method in methods:
                        z = diffusion_predict(model, packed, noise.clone(), method)
                        units = codec.decode(z.reshape(-1, 48) * cache.scale + mean)
                        for i, case in enumerate(group):
                            native = units[i * 16]; predicted = b''.join(units[i * 16:i * 16 + case['block_length']])[:32]
                            row = {'id': case['id'], 'window': case['window'], 'seed': seed, 'predicted_hex': predicted.hex(),
                                   'native_unit_hex': native.hex(), **byte_score(bytes.fromhex(case['target_hex']), bytes.fromhex(case['reference_hex']), predicted, native)}
                            rows[method].append(row); handles[method].write(json.dumps(row) + '\n')
                if (begin // args.batch) % 16 == 0: print(json.dumps({'model': name, 'anchors_done': begin + len(group), 'total': len(cases)}), flush=True)
            for method in methods:
                handles[method].close(); save_group(name + '_' + method, rows[method], {'checkpoint': path, 'state_step': state['state']['step'], 'weights': 'ema', 'sampler': method})
            del model, state, packed
            torch.cuda.empty_cache()
        del codec
    if args.only != 'diffusion':
        model = AutoModelForCausalLM.from_pretrained(BASELINE, local_files_only=True, torch_dtype=torch.float32).cuda().eval()
        assert sum(p.numel() for p in model.parameters()) == 20537568
        for greedy in (False, True):
            label = 'ar_greedy' if greedy else 'ar_sample'; rows = []
            with (folder / f'{label}.jsonl').open('x', encoding='utf-8', buffering=1) as handle:
                for seed in ([0] if greedy else [42, 123]):
                    for begin in range(0, len(cases), args.batch):
                        group = cases[begin:begin + args.batch]
                        predicted, ended, healed = ar_predict(model, tokenizer, byte_vocab, group, seed + begin * 1009, greedy)
                        for i, case in enumerate(group):
                            row = {'id': case['id'], 'window': case['window'], 'seed': seed, 'predicted_hex': predicted[i][:32].hex(),
                                   'eos': ended[i], 'known_healing_bytes': healed[i],
                                   **byte_score(bytes.fromhex(case['target_hex']), bytes.fromhex(case['reference_hex']), predicted[i])}
                            rows.append(row); handle.write(json.dumps(row) + '\n')
                        if (begin // args.batch) % 16 == 0: print(json.dumps({'model': label, 'seed': seed, 'anchors_done': begin + len(group)}), flush=True)
            save_group(label, rows, {'model': str(BASELINE), 'parameters': 20537568, 'policy': 'greedy' if greedy else 'temperature1_untruncated'})
    report['elapsed_seconds'] = time.perf_counter() - started
    write_json(folder / ('summary.json' if not args.only else args.only + '-summary.json'), report)


if __name__ == '__main__':
    main()
