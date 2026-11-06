"""Measure pipelined coarse scoring against recorded complete-document paths."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.ae_scoring_pipeline import ConcurrentAENLLScorer
from char_diff.ae_path_runtime import FrequentScoreCache
from char_diff.coarse_pipeline import solve_coarse_pipeline
from char_diff.segmented_cache import read_segmented_shard
from char_diff.gpu_monitor import GPUMonitor
from char_diff.utils import write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--shard',type=Path)
    p.add_argument('--batches',type=int,nargs='+',default=[32768]);p.add_argument('--workers',type=int,nargs='+',default=[1,4])
    p.add_argument('--group-documents',type=int,default=1024);p.add_argument('--group-bytes',type=int,default=1024**2)
    p.add_argument('--switch-interval',type=float,default=0.0005)
    p.add_argument('--cuda-streams',type=int,default=0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    sys.setswitchinterval(a.switch_interval)
    docs=[];reference=[]
    if a.shard:
        shard=read_segmented_shard(a.shard)
        for i in range(shard['manifest']['documents']):
            b,e=map(int,shard['byte_offsets'][i:i+2]);u,v=map(int,shard['unit_offsets'][i:i+2])
            docs.append({'raw':shard['text'][b:e]})
            reference.append([0]+np.cumsum(shard['lengths'][u:v]).astype(int).tolist())
    else:
        rows=json.loads(a.reference.read_text(encoding='utf8'))['variants']['8']['rows']
        docs=[{'raw':bytes.fromhex(r['source_hex'])} for r in rows];reference=[r['boundaries'] for r in rows]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    from char_diff.ae_path_runtime import _dag_path
    _dag_path(np.ones((2,1)))
    evidence={'documents':len(docs),'source_bytes':sum(len(d['raw']) for d in docs),'trials':[]}
    for batch in a.batches:
        for workers in a.workers:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            scorer=(ConcurrentAENLLScorer(ae,device='cuda',batch_size=batch,streams=a.cuda_streams) if a.cuda_streams
                    else OptimizedAENLLScorer(ae,device='cuda',batch_size=batch));scorer([b'warmup'])
            monitor=GPUMonitor(a.out/f'gpu_b{batch}_w{workers}.jsonl',interval=1).start()
            tick=time.perf_counter()
            try:
                results,timing=solve_coarse_pipeline(docs,scorer,workers=workers,group_documents=a.group_documents,
                                     group_bytes=a.group_bytes,score_cache=FrequentScoreCache(500000))
                row={'batch':batch,'workers':workers,'seconds':time.perf_counter()-tick,**timing,
                     'same_paths':sum(list(r['path'].boundaries)==ref for r,ref in zip(results,reference)),
                     'units':sum(len(r['path'].units) for r in results),'gpu':monitor.summary(),
                     'switch_interval':a.switch_interval,
                     'cuda_streams':a.cuda_streams,
                     'differences':[{'index':i,'source_hex':docs[i]['raw'].hex(),'reference':ref,
                                     'actual':list(r['path'].boundaries)}
                                    for i,(r,ref) in enumerate(zip(results,reference)) if list(r['path'].boundaries)!=ref]}
            except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:row={'batch':batch,'workers':workers,'error':str(exc)[:500]}
            finally:
                monitor.close()
                if hasattr(scorer,'close'):scorer.close()
            evidence['trials'].append(row);write_json(a.out/'results.json',evidence);print(json.dumps(row),flush=True)
