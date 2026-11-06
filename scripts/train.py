import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.training import train
from char_diff.utils import load_config

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("config",nargs="?",default="configs/tinystories_x0_22m.yaml")
    p.add_argument("--set",action="append",default=[])
    p.add_argument("--resume")
    p.add_argument("--fork-from",help="New continuation branch with full optimizer, EMA, data and RNG state")
    p.add_argument("--gmm-from",help="Expand a point x0 checkpoint to GMM; reset optimizer, retain data position and global progress")
    p.add_argument("--block-from",help="Change diffusion block size; retain all weights, EMA and data position, reset optimizer")
    p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--device")
    args=p.parse_args()
    train(load_config(args.config,args.set),args.resume,args.stop_after,args.device,args.fork_from,args.gmm_from,args.block_from)
