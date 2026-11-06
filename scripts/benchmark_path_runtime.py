"""Profile fixed AE stages, then benchmark compiled pipelined exact NLL paths."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ae_nll import AENLLScorer,tensorize_compact
from char_diff.ae_quality import targets_from_lengths
from char_diff.ae_path_runtime import FrequentScoreCache,solve_stream
from char_diff.ae_paths import candidate_spans
from char_diff.utils import sha256,write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


@torch.inference_mode()
def profile_stages(ae,units):
    ae.eval();ids,lens,mask=tensorize_compact(units,ae.l_max,'cuda')
    targets=targets_from_lengths(ids,lens);valid=targets>=0
    ae(ids,mask);torch.cuda.synchronize();trials=[]
    for _ in range(3):
        events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record();z=ae.encoder(ids,mask);events[1].record()
        logits=ae.decoder(z.repeat_interleave(33,0),torch.arange(33,device='cuda').repeat(len(units))).view(len(units),33,257)
        events[2].record();logp=logits.log_softmax(-1)
        nll=-logp.gather(-1,targets.clamp_min(0)[...,None]).squeeze(-1).masked_fill(~valid,0).sum(-1)
        feasible=((logits.argmax(-1)==targets)|~valid).all(-1)
        result=torch.stack((feasible.float(),nll),-1);events[3].record()
        result.cpu();torch.cuda.synchronize()
        trials.append({'encoder_ms':events[0].elapsed_time(events[1]),'decoder_ms':events[1].elapsed_time(events[2]),
                       'reduction_ms':events[2].elapsed_time(events[3]),'gpu_total_ms':events[0].elapsed_time(events[3])})
    return {'batch':len(units),'mean_bytes':float(np.mean(list(map(len,units)))),
            'median':{key:float(np.median([r[key] for r in trials])) for key in trials[0]},'trials':trials}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--documents',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--active-documents',type=int,default=64);p.add_argument('--cache-capacity',type=int,default=0)
    p.add_argument('--profile',action='store_true');p.add_argument('--load-cache',type=Path)
    p.add_argument('--optimized-scorer',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[json.loads(s) for s in a.documents.read_text(encoding='utf8').splitlines() if s]
    docs=[{'raw':bytes.fromhex(r['source_hex']) if 'source_hex' in r else r['text'].encode(),
           'language':r['language'],'doc_id':r.get('doc_id',r.get('id'))} for r in rows]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    if a.optimized_scorer:
        from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
        scorer=OptimizedAENLLScorer(ae,device='cuda',batch_size=a.batch_size)
    else:scorer=AENLLScorer(ae,device='cuda',batch_size=a.batch_size)
    scorer([b'warmup'])
    source_root=Path(__file__).resolve().parents[1]
    scope={'checkpoint':sha256(a.checkpoint),'precision':'fp32','scorer':sha256(source_root/'char_diff/ae_nll.py'),
           'optimized_scorer':sha256(source_root/'char_diff/ae_scoring_runtime.py') if a.optimized_scorer else None,
           'torch':torch.__version__,'device':torch.cuda.get_device_name()}
    cache=FrequentScoreCache(a.cache_capacity,scope)
    if a.load_cache:cache.load(a.load_cache)
    profile=None
    if a.profile:
        candidates=list(dict.fromkeys(d['raw'][s:e] for d in docs[:4] for s,e in candidate_spans(d['raw'])))
        profile=profile_stages(ae,candidates[:a.batch_size])
    def progress(item):
        progress.done+=1
        if progress.done%48==0:print(json.dumps({'documents':progress.done,'of':len(docs)}),flush=True)
    progress.done=0
    results,timing=solve_stream(docs,scorer,active_documents=a.active_documents,batch_size=a.batch_size,
                                score_cache=cache,on_result=progress)
    records=[];same=0
    with (a.out/'paths.jsonl').open('w',encoding='utf8') as handle:
        for item in results:
            row=rows[item['index']];doc=item['row'];path=item['path'];reference=row.get('boundaries')
            matched=list(path.boundaries)==reference if reference is not None else None
            same+=bool(matched)
            record={'doc_id':doc['doc_id'],'language':doc['language'],'source_hex':doc['raw'].hex(),
                    'source_bytes':len(doc['raw']),'boundaries':list(path.boundaries),'cost':path.cost,
                    'total_nll':path.quality_sum,'k':len(path.units),'coverage':True,'same_path':matched,
                    'units':[{'hex':u.hex(),'bytes':len(u)} for u in path.units],'rounds':item['rounds']}
            records.append(record);handle.write(json.dumps(record,separators=(',',':'))+'\n')
    if a.cache_capacity:cache.save(a.out/'scores.bin')
    evidence={'scope':scope,'input_sha256':sha256(a.documents),'documents':len(docs),
              'source_bytes':sum(len(d['raw']) for d in docs),'units':sum(r['k'] for r in records),
              'same_paths':same if any('boundaries' in r for r in rows) else None,
              'batch_size':a.batch_size,'active_documents':a.active_documents,'profile':profile,'timing':timing}
    write_json(a.out/'results.json',evidence);print(json.dumps(evidence),flush=True)
    if any('boundaries' in r for r in rows):assert same==len(docs),'Investigate changed segmentation'


if __name__=='__main__':main()
