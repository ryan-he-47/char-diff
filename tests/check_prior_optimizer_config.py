"""Regression for the deployed YAML -> JSON -> AdamW epsilon type."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.utils import ROOT, load_config, write_json
from char_diff.prior_training import byte_lr

rows = []
for arm in ('bpe', 'grid8'):
    cfg = load_config(ROOT / f'configs/multilingual_x0_78m_{arm}.yaml')
    # Follow deployment -> actual training loader, rather than json.loads.
    path = ROOT / f'runs/multilingual_prior_build_20261106/config_loader_r2/{arm}.json'
    write_json(path, cfg)
    loaded = load_config(path)
    assert loaded == cfg
    assert load_config(path, ['train.micro_batch=32'])['train']['micro_batch'] == 32
    tc = loaded['train']
    assert isinstance(tc['eps'], float) and tc['eps'] == 1e-8
    assert isinstance(tc['min_lr'], float) and tc['min_lr'] == 3e-6
    assert math.isclose(byte_lr(1000, 1000, tc), tc['min_lr'], rel_tol=1e-12)
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=tc['lr'], betas=tuple(tc['betas']),
        eps=tc['eps'], weight_decay=tc['weight_decay'], foreach=True)
    parameter.square().sum().backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    rows.append(dict(arm=arm, eps=tc['eps'], eps_type=type(tc['eps']).__name__,
                     min_lr=tc['min_lr'], loader='load_config(written production.json)',
                     adamw_step='finite', cooldown_endpoint='correct'))
write_json(ROOT / 'runs/multilingual_prior_build_20261106/optimizer_config_r2.json', rows)
print(json.dumps(rows))
