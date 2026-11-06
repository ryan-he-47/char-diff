"""Compare the deployed thread pipeline with process/array scoring at 4/8 CPUs."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='1'
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--shard',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8192)
    p.add_argument('--cuda-streams',type=int,default=4)
    p.add_argument('--case',choices=['threads4','process4','process8'])
    p.add_argument('--cases',nargs='+',default=['threads4','process4','process8'])
    p.add_argument('--cache-backend',choices=['python','compiled'],default='python')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.case is None:
        results=[]
        for case in a.cases:
            subprocess.run([sys.executable,'-u',__file__,'--checkpoint',str(a.checkpoint),'--shard',str(a.shard),
                '--out',str(a.out),'--batch-size',str(a.batch_size),'--cuda-streams',str(a.cuda_streams),'--case',case,
                '--cache-backend',a.cache_backend],check=True)
            results.append(json.loads((a.out/(case+'.json')).read_text(encoding='utf8')))
        from char_diff.utils import write_json
        write_json(a.out/'results.json',results);return
    cpus=8 if a.case=='process8' else 4
    affinity=None
    if hasattr(os,'sched_getaffinity'):
        available=sorted(os.sched_getaffinity(0))
        if len(available)<cpus:raise RuntimeError(f'Expected {cpus} allocated CPUs, found {available}')
        affinity=available[:cpus];os.sched_setaffinity(0,affinity)
    import numpy as np
    import torch
    from char_diff.ae_path_runtime import FrequentScoreCache,_dag_path
    from char_diff.coarse_pipeline import solve_coarse_pipeline
    from char_diff.coarse_process import ProcessCoarseSolver
    from char_diff.ae_scoring_pipeline import ConcurrentAENLLScorer
    from char_diff.ae_scoring_array import ConcurrentPackedScorer
    from char_diff.compiled_score_cache import CompiledScoreCache
    from char_diff.segmented_cache import read_segmented_shard
    from char_diff.gpu_monitor import GPUMonitor
    from char_diff.utils import write_json
    from char_diff.vendor.bltz.config import Cfg
    from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
    torch.set_num_threads(4 if a.case=='threads4' else 1);torch.backends.cuda.matmul.allow_tf32=False
    sys.setswitchinterval(.0005)
    shard=read_segmented_shard(a.shard);docs=[];reference=[]
    for i in range(shard['manifest']['documents']):
        b,e=map(int,shard['byte_offsets'][i:i+2]);u,v=map(int,shard['unit_offsets'][i:i+2])
        docs.append({'raw':shard['text'][b:e]});reference.append([0]+np.cumsum(shard['lengths'][u:v]).astype(int).tolist())
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    _dag_path(np.ones((2,1)));pool=None
    if a.case!='threads4':pool=ProcessCoarseSolver(cpus-2)
    cls=ConcurrentAENLLScorer if a.case=='threads4' else ConcurrentPackedScorer
    scorer=cls(ae,device='cuda',batch_size=a.batch_size,streams=a.cuda_streams);scorer([b'warmup'])
    cache=CompiledScoreCache(500000) if a.cache_backend=='compiled' else FrequentScoreCache(500000)
    torch.cuda.reset_peak_memory_stats();monitor=GPUMonitor(a.out/(a.case+'_gpu.jsonl'),interval=1).start()
    start=time.perf_counter()
    try:
        args={'score_cache':cache,'group_documents':1024,'group_bytes':1024**2}
        if pool:rows,timing=pool.solve(docs,scorer,**args)
        else:rows,timing=solve_coarse_pipeline(docs,scorer,workers=4,**args)
        paths=[list(r['path'].boundaries) for r in rows]
        row={'case':a.case,'cpus':cpus,'affinity':affinity,'batch_size':a.batch_size,'cuda_streams':a.cuda_streams,'cache_backend':a.cache_backend,
             'documents':len(docs),'source_bytes':sum(len(d['raw']) for d in docs),'total_seconds':time.perf_counter()-start,
             'timing':timing,'gpu':monitor.summary(),'same_paths':sum(a==b for a,b in zip(paths,reference)),
             'units':sum(len(r['path'].units) for r in rows),
             'differences':[{'index':i,'source_hex':docs[i]['raw'].hex(),'reference':ref,'actual':path}
                            for i,(path,ref) in enumerate(zip(paths,reference)) if path!=ref]}
        write_json(a.out/(a.case+'.json'),row);write_json(a.out/(a.case+'_paths.json'),paths)
        print(json.dumps({k:v for k,v in row.items() if k!='differences'}),flush=True)
    finally:
        monitor.close();scorer.close()
        if pool:pool.close()


if __name__=='__main__':main()
