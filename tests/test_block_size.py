"""Targeted checks for cache reblocking, weight inheritance and exact branch recovery."""
import copy
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.data import LatentCache
from char_diff.model import CLEAN, NOISY, BOS
from char_diff.training import train
from char_diff.utils import ROOT, load_config, write_json, sha256


def check_layout():
    path=ROOT/'data/tiny_smoke'; base=LatentCache(path); original=base.batch([0,1])
    result={}
    for block in [1,4]:
        cache=LatentCache(path,block=block); actual=cache.batch([0,1])
        assert cache.fingerprint==base.fingerprint and len(cache)==len(base)
        assert cache.target_units([0,1])==base.target_units([0,1])
        for key in ['x','roles','positions','docs']:
            assert torch.equal(actual[key],original[key]),key
        for b in range(2):
            roles=actual['roles'][b]; pos=actual['positions'][b]; docs=actual['docs'][b]
            clean=(roles==CLEAN)|(roles==BOS); noisy=roles==NOISY
            for q in torch.where(noisy)[0].tolist():
                allowed=actual['allowed'][b,0,q]
                start=int(actual['starts'][b,q]); length=int(actual['lengths'][b,q])
                assert 1<=length<=block and length==min(block,int(pos[noisy & (docs==docs[q])].max())-start+1)
                expected_history=clean & (docs==docs[q]) & (pos<start)
                expected_block=noisy & (docs==docs[q]) & (pos>=start) & (pos<start+length)
                assert torch.equal(allowed,expected_history|expected_block)
                assert not allowed[clean & (docs==docs[q]) & (pos>=start)].any()
                if block==1: assert int((allowed & noisy).sum())==1
        result[str(block)]={'target_units':cache.target_units([0,1]),'physical_positions':actual['roles'].numel(),
                            'same_windows_targets_and_clean_inputs':True,'future_clean_hidden':True}
    return result


def check_recovery():
    cfg=load_config(ROOT/'configs/tinystories_x0_block1.yaml')
    cfg['data'].update(cache='data/tiny_smoke',block=16)
    cfg['model'].update(width=64,layers=2,heads=4,ffn=128,block=16,grad_checkpoint=False)
    cfg['train'].update(batch_windows=2,micro_batch=1,precision='fp32',max_steps=2,log_every=99,save_every=99)
    cfg['train'].pop('reference_schedule');cfg['train'].pop('block_source_sha256')
    cfg['storage']['require_quota']=False
    with tempfile.TemporaryDirectory(prefix='block_size_check_',dir=ROOT/'runs') as temporary:
        folder=Path(temporary).resolve();folder.relative_to((ROOT/'runs').resolve())
        cfg['train']['run_dir']=str(folder/'parent')
        parent=train(cfg,device='cpu')['checkpoint']; source=torch.load(parent,weights_only=False)
        lite=folder/'weights.pt'
        torch.save({k:v for k,v in source.items() if k not in ['optimizer','scaler','sampler','rng','norms']},lite)
        cache=LatentCache(ROOT/'data/tiny_smoke'); schedule=folder/'schedule.json'
        write_json(schedule,dict(epoch_units=cache.units,total_steps=5,total_seen_units=cache.units*3,
                                 learning_rates=[.0003,.0003,0.,.0003,.00015]))
        cfg['model']['block']=cfg['data']['block']=1
        cfg['train'].update(max_steps=0,reference_schedule=str(schedule),block_source_sha256=sha256(lite),run_dir=str(folder/'full'))
        full=train(cfg,block_from=lite,device='cpu')['checkpoint']
        cfg['train']['run_dir']=str(folder/'resumed')
        partial=train(cfg,block_from=lite,stop_after=1,device='cpu')['checkpoint']
        inherited=torch.load(partial,weights_only=False)
        # The first branch LR is zero, making exact source-weight inheritance observable.
        for key,value in source['model'].items():
            assert torch.equal(value,inherited['model'][key]),key
            expected=source['ema'][key].clone().lerp_(value,1-cfg['train']['ema'])
            torch.testing.assert_close(inherited['ema'][key],expected,atol=0,rtol=0)
        assert inherited['model']['lengths.weight'].shape==(17,64)
        final=train(cfg,resume=partial,device='cpu')['checkpoint']
        a=torch.load(full,weights_only=False);b=torch.load(final,weights_only=False)
        assert a['state']==b['state'] and a['state']['step']==5
        assert a['sampler']==b['sampler'] and a['sampler']['cursor']==10
        for kind in ['model','ema']:
            for key in a[kind]: assert torch.equal(a[kind][key],b[kind][key]),(kind,key)
        assert all(int(v['step'])==3 for v in a['optimizer']['state'].values())
        rows=[json.loads(line) for line in (folder/'full/train.jsonl').read_text().splitlines()]
        assert [r['lr'] for r in rows]==[0.,.0003,.00015]
    return {'weights_and_ema_inherited':True,'original_length_rows_preserved':True,
            'exact_resumed_model_ema_cursor':True,'global_schedule_and_endpoint':True}


if __name__=='__main__':
    torch.set_num_threads(2)
    result={'cache_reblocking':check_layout(),'warm_start_recovery':check_recovery()}
    write_json(ROOT/'docs/experiments/block-size/evidence/targeted-checks.json',result)
    print(json.dumps(result),flush=True)
