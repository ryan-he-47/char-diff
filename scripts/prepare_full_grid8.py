"""Prepare heldout exclusions, then stream the pinned full corpus through grid8."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from char_diff.full_corpus import CorpusIndex,segment_corpus
from char_diff.storage import StorageGuard
from char_diff.utils import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['index','run'],required=True)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--heldout-hashes',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--expected-checkpoint-sha256',
                   default='26df106186699b7ef13a0d1c4db096e107a701d7fb4e71c2621d99c3e5d6ea61')
    p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--group-documents',type=int,default=64)
    p.add_argument('--group-bytes',type=int,default=262144)
    p.add_argument('--shard-bytes',type=int,default=8*1024**2)
    p.add_argument('--cache-capacity',type=int,default=500000)
    p.add_argument('--pipeline-workers',type=int,default=0)
    p.add_argument('--cpu-processes',type=int,default=0)
    p.add_argument('--cuda-streams',type=int,default=0)
    p.add_argument('--upgrade-runtime',action='store_true')
    p.add_argument('--max-shards',type=int,default=0)
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--project-limit-gib',type=float,default=32)
    p.add_argument('--require-quota',action='store_true')
    a=p.parse_args();a.metrics.mkdir(parents=True,exist_ok=True)
    if min(a.batch_size,a.group_documents,a.group_bytes,a.shard_bytes)<1:p.error('Positive capacities required')
    if min(a.cpu_processes,a.pipeline_workers,a.cuda_streams)<0:p.error('Worker counts must be nonnegative')
    if a.cpu_processes and a.pipeline_workers:p.error('Choose CPU processes or CPU preparation threads')
    guard=StorageGuard(a.project_root,limit_gib=a.project_limit_gib,reserve_gib=1,require_quota=a.require_quota)
    source=json.loads(a.source_manifest.read_text(encoding='utf8'))
    guard.check(512*1024**2)
    index=CorpusIndex(a.out,source,a.heldout_hashes.read_bytes())
    if a.phase=='index':
        def log_file(done,entry):
            event={'phase':'excluded','files_completed':done,'language':entry['language'],'time':time.time()}
            write_json(a.metrics/'index_progress.json',event);print(json.dumps(event),flush=True)
        count=index.prepare_exclusions(log_file)
        write_json(a.metrics/'index_result.json',{'status':'complete','excluded_documents':count,
                   'source_manifest_sha256':sha256(a.source_manifest),'storage':guard.check(512*1024**2)})
        index.close();return 0

    if a.checkpoint is None:p.error('--checkpoint is required for run')
    if sha256(a.checkpoint)!=a.expected_checkpoint_sha256:
        raise ValueError('Checkpoint differs from the pinned segmentation AE')
    import torch
    from char_diff.ae_path_runtime import CompiledLazyGraph
    from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
    from char_diff.ae_scoring_pipeline import ConcurrentAENLLScorer
    from char_diff.ae_scoring_array import ConcurrentPackedScorer
    from char_diff.coarse_process import ProcessCoarseSolver
    from char_diff.gpu_monitor import GPUMonitor
    from char_diff.vendor.bltz.config import Cfg
    from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
    torch.set_num_threads(1 if a.cpu_processes else 4);torch.backends.cuda.matmul.allow_tf32=False
    if a.pipeline_workers:sys.setswitchinterval(0.0005)
    root=Path(__file__).resolve().parents[1]
    files=['char_diff/ae_nll.py','char_diff/ae_quality.py','char_diff/ae_paths.py','char_diff/ae_lazy_paths.py',
           'char_diff/ae_path_runtime.py','char_diff/ae_scoring_runtime.py','char_diff/ae_scoring_pipeline.py','char_diff/path_proposals.py',
           'char_diff/coarse_paths.py','char_diff/coarse_pipeline.py','char_diff/coarse_process.py',
           'char_diff/ae_scoring_array.py','char_diff/compiled_score_cache.py','char_diff/full_corpus.py','char_diff/segmented_cache.py',
           'char_diff/vendor/bltz/models/autoencoder.py','char_diff/vendor/bltz/models/encoder.py',
           'char_diff/vendor/bltz/models/head.py','scripts/prepare_full_grid8.py']
    scoring={'checkpoint_sha256':sha256(a.checkpoint),'precision':'fp32','tf32':False,
             'torch':torch.__version__,'device':torch.cuda.get_device_name(),'batch_size':a.batch_size,
             'cuda_streams':a.cuda_streams,
             'code':{name:sha256(root/name) for name in files}}
    provenance={'source_manifest_sha256':sha256(a.source_manifest),'source_revision':source['revision'],
                'languages':source['languages'],'old_holdout_sha256':sha256(a.heldout_hashes),
                'exclusions':'all upstream test + old calibration/test','deduplication':'global exact UTF8 bytes',
                'scoring':scoring,'proposal':'grid8','solver':'eager','quality_weight':10.,
                'quality':'byte+EOS total NLL','feasibility':'exact byte+EOS argmax','max_unit_bytes':32,
                'boundaries':'UTF8 codepoints','group_documents':a.group_documents,
                'group_bytes':a.group_bytes,'shard_bytes':a.shard_bytes,'cache_capacity':a.cache_capacity,
                'pipeline_workers':a.pipeline_workers,'cpu_processes':a.cpu_processes}
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    process_solver=ProcessCoarseSolver(a.cpu_processes) if a.cpu_processes else None
    scorer=(ConcurrentPackedScorer(ae,device='cuda',batch_size=a.batch_size,streams=a.cuda_streams or 1) if a.cpu_processes
            else ConcurrentAENLLScorer(ae,device='cuda',batch_size=a.batch_size,streams=a.cuda_streams) if a.cuda_streams
            else OptimizedAENLLScorer(ae,device='cuda',batch_size=a.batch_size));scorer([b'warmup'])
    CompiledLazyGraph(b'a').optimistic_path();torch.cuda.reset_peak_memory_stats()
    monitor=GPUMonitor(a.metrics/'gpu_samples.jsonl').start();started=time.perf_counter()
    initial=index.get('state');stop=False;last_event=0.;last_guard=time.monotonic()
    def request_stop(signum,frame):
        nonlocal stop
        stop=True;print(json.dumps({'event':'stop_requested','signal':signum}),flush=True)
    for name in ('SIGTERM','SIGUSR1'):
        if hasattr(signal,name):signal.signal(getattr(signal,name),request_stop)
    def check_storage():
        nonlocal last_guard
        if time.monotonic()-last_guard>=300:
            guard.check(512*1024**2);last_guard=time.monotonic()
    def emit(kind,current,stats=None):
        nonlocal last_event
        last_event=time.monotonic()
        totals={key:sum(row[key] for row in current['counts'].values()) for key in next(iter(current['counts'].values()))}
        event={'event':kind,'timestamp':time.time(),'pid':os.getpid(),'cursor':current['cursor'],
               'shards':current['shards'],'complete':current['complete'],'totals':totals,
               'invocation_seconds':time.perf_counter()-started,'gpu':monitor.summary(),'statistics':stats or {}}
        with (a.metrics/'events.jsonl').open('a',encoding='utf8') as handle:handle.write(json.dumps(event)+'\n')
        write_json(a.metrics/'runtime.json',event);print(json.dumps(event),flush=True)
    def on_group(current,stats):
        if time.monotonic()-last_event>=30:emit('working',current,stats)
    emit('start',initial)
    try:
        result=segment_corpus(index,scorer,provenance,batch_size=a.batch_size,cache_capacity=a.cache_capacity,
            group_documents=a.group_documents,group_bytes=a.group_bytes,shard_bytes=a.shard_bytes,
            max_shards=a.max_shards,on_group=on_group,
            on_commit=lambda event:emit('committed',event['state'],event['statistics']),
            guard=check_storage,should_stop=lambda:stop,pipeline_workers=a.pipeline_workers,
            upgrade_runtime=a.upgrade_runtime,process_solver=process_solver)
        if result['complete']:
            actual={lang:row['scanned'] for lang,row in result['counts'].items()}
            if actual!=source['expected_train_documents']:raise ValueError('Scanned rows differ from pinned dataset card')
        emit('complete' if result['complete'] else 'checkpointed',result)
        return 0 if result['complete'] or a.max_shards else 75
    finally:
        monitor.close();index.close()
        if hasattr(scorer,'close'):scorer.close()
        if process_solver is not None:process_solver.close()


if __name__=='__main__':raise SystemExit(main())
