"""Fixed-width x0 mixtures and local posterior integration for the linear flow.

Packed output: K logits followed by K contiguous D-dimensional clean means.
The posterior ratio follows GMFlow (arXiv:2504.05304, Eq. 10). All density
and integration arithmetic uses FP32; the transformer may run under AMP.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def unpack(prediction, dim=48):
    if prediction.shape[-1] % (dim + 1):
        raise ValueError("GMM output must contain K logits and K clean vectors")
    k = prediction.shape[-1] // (dim + 1)
    if k < 1:
        raise ValueError("GMM requires at least one component")
    value = prediction.float()
    return value[..., :k], value[..., k:].reshape(*value.shape[:-1], k, dim)


def mixture_mean(prediction, dim=48):
    logits, means = unpack(prediction, dim)
    return (logits.softmax(-1)[..., None] * means).sum(-2)


def _log_terms(prediction, target, sigma):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("GMM sigma must be finite and positive")
    logits, means = unpack(prediction, target.shape[-1])
    squared = (means - target.float().unsqueeze(-2)).square().sum(-1)
    log_pi = F.log_softmax(logits, -1)
    return log_pi, log_pi - squared / (2 * sigma * sigma)


def loss_sum(prediction, batch, sigma):
    # Same unnormalized D-dimensional convention as flow.loss_sum.
    # training.py divides by D * the complete accumulated batch target count.
    from .model import NOISY
    valid = batch["roles"] == NOISY
    _, scores = _log_terms(prediction[valid], batch["x"][valid], sigma)
    return (-2 * sigma * sigma * torch.logsumexp(scores, -1)).sum(), int(valid.sum().item())


@torch.no_grad()
def diagnostics(prediction, batch, sigma):
    from .model import NOISY
    valid = batch["roles"] == NOISY
    log_pi, scores = _log_terms(prediction[valid], batch["x"][valid], sigma)
    log_r = scores.log_softmax(-1)
    mass = log_r.exp().mean(0)
    return {
        "gmm_probe_units": int(valid.sum()),
        "gmm_posterior_effective_mean": float((-(log_r.exp() * log_r).sum(-1)).exp().mean()),
        "gmm_gate_effective_mean": float((-(log_pi.exp() * log_pi).sum(-1)).exp().mean()),
        "gmm_posterior_mass": mass.tolist(),
    }


def posterior_at(prediction, source_z, target_z, source_t, target_t, sigma):
    """q(x|target_z) proportional to q(x|source_z) p(target_z|x)/p(source_z|x).

    Shared spherical component variance makes the determinant correction common
    to all weights. Factoring the precision difference avoids cancellation.
    target_t stays strictly positive; the integrator never evaluates at zero.
    """
    if not 0 < target_t < source_t <= 1:
        raise ValueError("Posterior conversion requires 0 < target_t < source_t <= 1")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("GMM sigma must be finite and positive")
    logits, means = unpack(prediction, source_z.shape[-1])
    delta = (source_t - target_t) * (source_t + target_t - 2 * source_t * target_t)
    ratio_variance = (source_t * target_t) ** 2 / delta
    ratio_center = ((1 - target_t) * source_t**2 * target_z.float()
                    - (1 - source_t) * target_t**2 * source_z.float()) / delta
    variance = sigma * sigma
    total = ratio_variance + variance
    centers = (ratio_variance * means + variance * ratio_center.unsqueeze(-2)) / total
    log_weights = (logits.log_softmax(-1)
                   - (means - ratio_center.unsqueeze(-2)).square().sum(-1) / (2 * total))
    return centers, log_weights.log_softmax(-1), variance * ratio_variance / total


def ode_step(prediction, z, t, next_t, sigma, substeps=4):
    """One network evaluation, followed by local posterior Euler substeps."""
    if not 0 <= next_t < t <= 1 or substeps < 1:
        raise ValueError("Require descending flow time and positive substeps")
    anchor = z.float()
    current = anchor
    for j in range(substeps):
        u = t + (next_t - t) * j / substeps
        v = t + (next_t - t) * (j + 1) / substeps
        if j == 0:
            clean = mixture_mean(prediction, z.shape[-1])
        else:
            means, log_weights, _ = posterior_at(prediction, anchor, current, t, u, sigma)
            clean = (log_weights.exp().unsqueeze(-1) * means).sum(-2)
        current = (v / u) * current + (1 - v / u) * clean
    return current


def expand_output_state(state, components, init_rms, seed):
    """Copy a point head to K means; preserve its initial mixture expectation."""
    if components < 1 or not math.isfinite(init_rms) or init_rms < 0:
        raise ValueError("Invalid GMM initialization")
    weight, bias = state["output.weight"], state["output.bias"]
    dim, width = weight.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    offsets = torch.zeros(components, dim, dtype=torch.float32)
    if components > 1 and init_rms:
        offsets = torch.randn(components, dim, generator=generator)
        offsets -= offsets.mean(0, keepdim=True)
        offsets *= init_rms / offsets.square().mean().sqrt()
    result = dict(state)
    result["output.weight"] = torch.cat([
        weight.new_zeros(components, width), weight.repeat(components, 1)])
    result["output.bias"] = torch.cat([
        bias.new_zeros(components), (bias[None] + offsets.to(bias)).flatten()])
    return result
