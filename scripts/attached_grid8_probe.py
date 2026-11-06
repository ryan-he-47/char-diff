"""Temporarily suspend the owned producer while profiling on its allocation."""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid',type=int,required=True);p.add_argument('--job',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--reference')
p.add_argument('--kind',choices=['batches','pipeline','profile'],default='batches');p.add_argument('--shard')
p.add_argument('--batches',type=int,nargs='+',default=[8192,32768,49152]);p.add_argument('--workers',type=int,nargs='+',default=[4])
p.add_argument('--cuda-streams',type=int,default=0)
p.add_argument('--out',required=True);a=p.parse_args()
environment=Path(f'/proc/{a.pid}/environ').read_bytes().split(b'\0')
command=Path(f'/proc/{a.pid}/cmdline').read_bytes()
if f'SLURM_JOB_ID={a.job}'.encode() not in environment or b'prepare_full_grid8.py' not in command:
    raise RuntimeError('Target PID is not the expected owned corpus job')
def interrupted(signum,frame):raise SystemExit(128+signum)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
script=Path(__file__).with_name({'batches':'probe_grid8_batches.py','pipeline':'benchmark_coarse_pipeline.py',
                               'profile':'profile_grid8_scorer.py'}[a.kind])
arguments=[sys.executable,'-u',str(script),'--checkpoint',a.checkpoint,'--out',a.out]
if a.kind=='batches':arguments+=['--reference',a.reference]
elif a.kind=='pipeline':arguments+=['--shard',a.shard,'--batches',*map(str,a.batches),'--workers',*map(str,a.workers),
                                   '--cuda-streams',str(a.cuda_streams)]
print(json.dumps({'event':'temporary_probe_pause','target_pid':a.pid,'job':a.job}),flush=True)
try:
    os.kill(a.pid,signal.SIGSTOP)
    subprocess.run(arguments,check=True,timeout=300)
finally:
    os.kill(a.pid,signal.SIGCONT)
    print(json.dumps({'event':'producer_resumed','target_pid':a.pid}),flush=True)
