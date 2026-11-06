"""Measure exact lazy NLL path solving against cached exhaustive reference paths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_lazy_paths import LazyNLLGraph
from char_diff.ae_nll import AENLLScorer
from char_diff.utils import write_json, sha256
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference-paths',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--device',default='cuda')
    p.add_argument('--compiled',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    rows=[json.loads(s) for s in a.reference_paths.read_text(encoding='utf8').splitlines() if s]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    scorer=AENLLScorer(ae,device=a.device,batch_size=a.batch_size)
    scorer([b'warmup'])
    Graph=LazyNLLGraph
    if a.compiled:
        from char_diff.ae_path_runtime import CompiledLazyGraph, _dag_path
        import numpy as np
        _dag_path(np.ones((2,1),np.float64))
        Graph=CompiledLazyGraph
    start=time.perf_counter();graphs=[Graph(bytes.fromhex(r['source_hex'])) for r in rows]
    graph_seconds=time.perf_counter()-start; active=set(range(len(rows))); solved={}; cache={}
    rounds=[];score_seconds=0.;dp_seconds=0.
    for iteration in range(10000):
        tick=time.perf_counter();pending={}
        for i in sorted(active):
            result, unknown=graphs[i].optimistic_path()
            if result is not None:
                solved[i]=result
            else:
                pending.update(dict.fromkeys(unknown))
        active.difference_update(solved)
        dp_seconds+=time.perf_counter()-tick
        if not active:break
        fresh=[u for u in pending if u not in cache]
        if not fresh: raise RuntimeError('Lazy graph made no progress')
        tick=time.perf_counter();scores=dict(zip(fresh,scorer(fresh)));cache.update(scores)
        score_seconds+=time.perf_counter()-tick
        for i in active: graphs[i].update(scores)
        rounds.append({'round':iteration+1,'active_docs':len(active),'new_candidates':len(fresh)})
        if (iteration+1)%20==0:print(json.dumps({**rounds[-1],'seconds':time.perf_counter()-start}),flush=True)
    if active:raise RuntimeError('Lazy iteration limit exceeded')
    records=[]
    for i,r in enumerate(rows):
        result=solved[i]
        records.append({'doc_id':r['doc_id'],'language':r['language'],'source_bytes':r['source_bytes'],
                        'same_path':list(result.boundaries)==r['boundaries'],
                        'units':len(result.units),'cost_error':result.cost-r['cost'],
                        'all_candidates':len(graphs[i].occurrences),'known_candidates':len(graphs[i].known)})
    result={'checkpoint_sha256':sha256(a.checkpoint),'reference_sha256':sha256(a.reference_paths),
            'compiled':a.compiled,
            'precision':'fp32','batch_size':a.batch_size,'device':str(scorer.device),
            'documents':len(rows),'same_paths':sum(r['same_path'] for r in records),
            'source_bytes':sum(r['source_bytes'] for r in records),'unique_scored_candidates':len(cache),
            'all_candidate_strings_across_documents':len(set(u for g in graphs for u in g.occurrences)),
            'rounds':rounds,'seconds':time.perf_counter()-start,'graph_seconds':graph_seconds,
            'score_seconds':score_seconds,'dp_seconds':dp_seconds,'rows':records}
    write_json(a.out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','rounds')}),flush=True)
    assert result['same_paths']==len(rows),'Review changed paths before deployment'


if __name__=='__main__':main()
