"""Local diagnostic of coarse candidate graphs; formal NLL10 graph stays unchanged."""
from __future__ import annotations
import argparse
import bisect
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_nll import AENLLScorer
from char_diff.ae_lazy_paths import LazyNLLGraph
from char_diff.ae_paths import boundary_offsets, NoFeasiblePath
from char_diff.path_proposals import proposed_boundaries
from char_diff.utils import write_json,sha256
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference-paths',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--strides',type=int,nargs='+',default=[8,16])
    p.add_argument('--modes',nargs='+')
    p.add_argument('--tokenizer',type=Path,default=Path('assets/tokenizer.json'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    scorer=AENLLScorer(ae,device='cuda',batch_size=512);scorer([b'warmup'])
    rows=[json.loads(s) for s in a.reference_paths.read_text(encoding='utf8').splitlines() if s]
    results={'purpose':'Local candidate proposal diagnostic only; not adopted for corpus preparation',
             'checkpoint_sha256':sha256(a.checkpoint),'reference_sha256':sha256(a.reference_paths),'variants':{}}
    modes=a.modes or [f'grid{stride}' for stride in a.strides]
    tokenizer=None
    if any(mode.startswith('bpe') for mode in modes):
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(a.tokenizer))
        results['tokenizer_sha256']=sha256(a.tokenizer)
    for mode in modes:
        tick=time.perf_counter();cache={};graphs=[];coarse=[];fallback_intervals=0
        for row in rows:
            raw=bytes.fromhex(row['source_hex']);full=boundary_offsets(raw)
            chosen=proposed_boundaries(raw,mode,tokenizer=tokenizer)
            coarse.append(chosen);graphs.append(LazyNLLGraph(raw,offsets=chosen))
        candidates=list(dict.fromkeys(u for g in graphs for u in g.occurrences))
        cache.update(zip(candidates,scorer(candidates)))
        # Expand only failed adjacent coarse spans to their complete UTF-8 nodes.
        # This preserves a covering route when individual characters reconstruct.
        for i,(g,chosen) in enumerate(zip(graphs,coarse)):
            full=boundary_offsets(g.raw);expanded=set(chosen)
            for left,right in zip(chosen,chosen[1:]):
                if not cache[g.raw[left:right]].feasible:
                    expanded.update(x for x in full if left<x<right);fallback_intervals+=1
            if len(expanded)!=len(chosen):graphs[i]=LazyNLLGraph(g.raw,offsets=sorted(expanded))
        extra=list(dict.fromkeys(u for g in graphs for u in g.occurrences if u not in cache))
        if extra:cache.update(zip(extra,scorer(extra)))
        records=[];total_units=0;cost=0.;nll=0.;covered_bytes=0
        for row,g in zip(rows,graphs):
            g.update({u:cache[u] for u in g.occurrences})
            try:
                result,pending=g.optimistic_path();assert not pending
                assert b''.join(result.units)==g.raw
                record={'doc_id':row['doc_id'],'language':row['language'],'coverage':True,'source_bytes':len(g.raw),
                        'units':len(result.units),'cost':result.cost,'total_nll':result.quality_sum,
                        'same_path':list(result.boundaries)==row['boundaries'],
                        'cost_increase':result.cost-row['cost'],'unique_candidates':len(g.occurrences),
                        'boundaries':list(result.boundaries),'source_hex':g.raw.hex(),
                        'units_hex':[u.hex() for u in result.units]}
                total_units+=len(result.units);cost+=result.cost;nll+=result.quality_sum;covered_bytes+=len(g.raw)
            except NoFeasiblePath as exc:
                record={'doc_id':row['doc_id'],'language':row['language'],'coverage':False,'error':str(exc)}
            records.append(record)
        ref_cost=sum(r['cost'] for r in rows)
        result={'mode':mode,'candidate_rule':'Grid or BPE original-text UTF8 boundaries; cap adjacent gaps at 32B; expand failed adjacent spans to full UTF8 nodes',
                'documents':len(rows),'covered_documents':sum(r['coverage'] for r in records),'units':total_units,
                'covered_bytes':covered_bytes,'bytes_per_unit':covered_bytes/total_units,'cost':cost,
                'cost_increase_fraction':cost/ref_cost-1,'total_nll':nll,'unique_scored_candidates':len(cache),
                'fallback_intervals':fallback_intervals,'seconds':time.perf_counter()-tick,'rows':records}
        key=mode[4:] if mode.startswith('grid') else mode
        results['variants'][key]=result;write_json(a.out/'results.json',results)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
