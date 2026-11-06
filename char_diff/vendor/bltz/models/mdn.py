"""MDN head for v2 (docs/31 §2.3): two-layer nonlinear buffer -> diagonal GMM.

Output target is a STRUCTURED GMM parameter space, not a flat vocab — the
buffer cushions the backbone from it (2026-09-18 拍板, mirrors the input-side
adapter; no control arm). Diagonal covariance only: each component encircles
ONE byte string's local region; cross-dim correlation is absorbed by component
multiplicity (Graves MDN / CoSE precedent). Full covariance at d=32 would
already cost ~27.6M in the output layer alone — rejected.

Shapes: h (..., d_in) -> logit_pi (..., K), mu (..., K, d), sigma (..., K, d).
Small-init final layer + zero bias => initial GMM ~ single wide Gaussian
(pi uniform, mu=0, sigma=1) — adaLN-Zero-flavored stable start.
Inference takes the component MEAN only (μ_k; 2026-09-17 拍板): temperature
acts on pi alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLU(nn.Module):
    """SwiGLU block (2026-09-24 对照实验臂): x -> W3(silu(W1 x) * W2 x)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_out)
        self.w2 = nn.Linear(d_in, d_out)
        self.w3 = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MDNHead(nn.Module):
    def __init__(
        self,
        d_in: int = 768,
        hidden: int = 1024,
        n_comp: int = 64,
        d_emb: int = 48,
        sigma_floor: float = 1e-3,
        swiglu: bool = False,
        depth: int = 2,
    ):
        super().__init__()
        self.n_comp = n_comp
        self.d_emb = d_emb
        self.sigma_floor = sigma_floor
        self.swiglu = swiglu
        self.depth = depth
        if depth >= 1:
            if swiglu:
                self.fc1 = _SwiGLU(d_in, hidden)
                self.fc2 = _SwiGLU(hidden, hidden) if depth >= 2 else None
            else:
                self.fc1 = nn.Linear(d_in, hidden)
                self.fc2 = nn.Linear(hidden, hidden) if depth >= 2 else None
        self.out = nn.Linear(hidden if depth >= 1 else d_in, n_comp * (1 + 2 * d_emb))
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def params(self, h: torch.Tensor):
        """h (..., d_in) -> logit_pi (..., K), mu (..., K, d), sigma (..., K, d)."""
        K, D = self.n_comp, self.d_emb
        if self.depth == 0:
            x = h
        elif self.depth == 1:
            x = self.fc1(h) if self.swiglu else F.silu(self.fc1(h))
        else:
            assert self.fc2 is not None, "depth>=2 requires fc2"
            x = self.fc2(self.fc1(h)) if self.swiglu else F.silu(self.fc2(F.silu(self.fc1(h))))
        o = self.out(x)
        logit_pi = o[..., :K]
        mu = o[..., K : K + K * D].view(*o.shape[:-1], K, D)
        sigma = F.elu(o[..., K + K * D :].view(*o.shape[:-1], K, D)) + 1.0 + self.sigma_floor
        return logit_pi, mu, sigma

    def nll(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-position NLL of target (..., d) under the GMM, fp32. -> (...,)."""
        logit_pi, mu, sigma = (t.float() for t in self.params(h))
        target = target.float()
        D = target.shape[-1]
        z = (target.unsqueeze(-2) - mu) / sigma  # (..., K, D)
        log_n = -0.5 * z.pow(2).sum(-1) - sigma.log().sum(-1) - 0.5 * D * math.log(2 * math.pi)
        return -torch.logsumexp(F.log_softmax(logit_pi, dim=-1) + log_n, dim=-1)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """h (..., d_in) -> mu of one component, (..., d). tau<=0 -> argmax pi."""
        logit_pi, mu, _sigma = self.params(h)
        if tau <= 0:
            k = logit_pi.argmax(dim=-1)
        else:
            k = torch.multinomial(F.softmax(logit_pi / tau, dim=-1).reshape(-1, self.n_comp), 1)
            k = k.view(h.shape[:-1])
        return mu.gather(-2, k.unsqueeze(-1).unsqueeze(-1).expand(*k.shape, 1, self.d_emb)).squeeze(-2)

    @torch.no_grad()
    def mode(self, h: torch.Tensor, iters: int = 24) -> torch.Tensor:
        """GMM mode (MAP point): argmax_x sum_k pi_k N(x; mu_k, sig_k).

        No closed form; solved by mean-shift-style fixed-point iteration on
        the stationarity condition grad log p = 0:
            x_d <- sum_k w_kd mu_kd / sum_k w_kd,  w_kd = r_k(x) / sig_kd^2
        Init at the densest component center (exact pairwise eval); converges
        deterministically in <32 iters (48-d, tiny compute). Note the mode is
        NOT argmax-pi's mu in general: center density weighs pi_k/prod(sigma),
        so sharp components dominate, and at overlapping variant clouds the
        mode can sit between centers.
        """
        logit_pi, mu, sig = self.params(h)
        K, D = self.n_comp, self.d_emb
        log_pi = F.log_softmax(logit_pi, dim=-1)  # (..., K)
        const = 0.5 * D * math.log(2 * math.pi)
        # density at each component center: log p(mu_i), (..., K, K) -> (..., K)
        z = (mu.unsqueeze(-2) - mu.unsqueeze(-3)) / sig.unsqueeze(-2)  # (...,K,K,D)
        log_n = -0.5 * z.pow(2).sum(-1) - sig.log().sum(-1).unsqueeze(-2) - const
        logp_at_mu = torch.logsumexp(log_pi.unsqueeze(-2) + log_n, dim=-1)
        x = mu.gather(-2, logp_at_mu.argmax(-1).unsqueeze(-1).unsqueeze(-1)
                      .expand(*logp_at_mu.shape[:-1], 1, D)).squeeze(-2).clone()
        for _ in range(iters):
            z = (x.unsqueeze(-2) - mu) / sig  # (..., K, D)
            log_n = -0.5 * z.pow(2).sum(-1) - sig.log().sum(-1) - const
            r = F.softmax(log_pi + log_n, dim=-1)  # (..., K)
            w = r.unsqueeze(-1) / sig.pow(2)       # (..., K, D)
            x_new = (w * mu).sum(-2) / w.sum(-2).clamp_min(1e-12)
            if float((x_new - x).abs().max()) < 1e-5:
                x = x_new
                break
            x = x_new
        return x

    @torch.no_grad()
    def modes(self, h: torch.Tensor, iters: int = 24, tol: float = 0.1):
        """ALL distinct local maxima + basin masses (2026-09-30 用户设计:
        多极大值点采样=词级温度的密度原生实现,无需词表).

        Multi-start mean-shift: run mode()'s fixed-point iteration from EVERY
        component center in parallel (K starts x K comps x D — trivial).
        Converged points are clustered (L2 < tol); basin mass of mode j =
        sum of softmax(pi) over components whose start converged into j
        (proper responsibility-weighted basin mass). Single position only
        (h: (..., d_in) -> modes (M, D), log_mass (M,), log_dens (M,),
        sorted by mass desc). Generation usage: sample j ~ softmax(log_mass/T)
        -> decode — every mode is on-manifold (mode readout champion), so
        sampled strings stay legal while allowing real diversity.
        """
        logit_pi, mu, sig = (t.float() for t in self.params(h))
        shape = logit_pi.shape[:-1]
        logit_pi = logit_pi.reshape(-1, self.n_comp)
        mu = mu.reshape(-1, self.n_comp, self.d_emb)
        sig = sig.reshape(-1, self.n_comp, self.d_emb)
        out = []
        for b in range(mu.shape[0]):
            lp, m_, s_ = logit_pi[b], mu[b], sig[b]        # (K), (K, D)
            K, D = m_.shape
            log_pi = F.log_softmax(lp, -1)
            const = 0.5 * D * math.log(2 * math.pi)
            x = m_.clone()                                  # K starts
            for _ in range(iters):
                z = (x.unsqueeze(1) - m_.unsqueeze(0)) / s_.unsqueeze(0)
                log_n = -0.5 * z.pow(2).sum(-1) - s_.log().sum(-1).unsqueeze(0) - const
                r = F.softmax(log_pi.unsqueeze(0) + log_n, dim=-1)  # (Ks, Kc)
                w = r.unsqueeze(-1) / s_.pow(2).unsqueeze(0)
                x_new = (w * m_.unsqueeze(0)).sum(1) / w.sum(1).clamp_min(1e-12)
                if float((x_new - x).abs().max()) < 1e-5:
                    x = x_new
                    break
                x = x_new
            # density at converged points
            z = (x.unsqueeze(1) - m_.unsqueeze(0)) / s_.unsqueeze(0)
            log_n = -0.5 * z.pow(2).sum(-1) - s_.log().sum(-1).unsqueeze(0) - const
            ld = torch.logsumexp(log_pi.unsqueeze(0) + log_n, dim=-1)  # (K starts,)
            # greedy clustering by density desc
            order = ld.argsort(descending=True)
            centers: list[torch.Tensor] = []
            assign = torch.full((K,), -1, dtype=torch.long, device=x.device)
            for s in order.tolist():
                if assign[s] >= 0:
                    continue
                d = (x - x[s]).norm(dim=-1)
                members = (d < tol) & (assign < 0)
                assign[members] = len(centers)
                centers.append(x[s])
            M = len(centers)
            modes = torch.stack(centers)
            log_mass = torch.log(torch.stack([
                log_pi[assign == j].exp().sum().clamp_min(1e-30) for j in range(M)]))
            # density at cluster centers (recompute cheaply)
            z = (modes.unsqueeze(1) - m_.unsqueeze(0)) / s_.unsqueeze(0)
            log_n = -0.5 * z.pow(2).sum(-1) - s_.log().sum(-1).unsqueeze(0) - const
            log_dens = torch.logsumexp(log_pi.unsqueeze(0) + log_n, dim=-1)
            m_order = log_mass.argsort(descending=True)
            out.append((modes[m_order], log_mass[m_order], log_dens[m_order]))
        return out[0] if len(out) == 1 else out
