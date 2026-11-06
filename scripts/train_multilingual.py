"""Train the approved full-corpus multilingual prior; exact resume is explicit."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.prior_training import train
from char_diff.utils import load_config

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('--resume'); p.add_argument('--device',default='cuda')
    p.add_argument('--stop-after',type=int,default=0); p.add_argument('--set',action='append',default=[])
    p.add_argument('--no-monitor',action='store_true')
    a=p.parse_args()
    result=train(load_config(a.config,a.set),resume=a.resume,stop_after=a.stop_after,device=a.device,monitor=not a.no_monitor)
    raise SystemExit(0 if result['event']=='complete' or a.stop_after else 75)
