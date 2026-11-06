import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.codec import Codec
from char_diff.cache import prepare
from char_diff.storage import StorageGuard
from char_diff.utils import ROOT

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="data/tinystories")
    p.add_argument("--device",default="cuda")
    p.add_argument("--train-jsonl")
    p.add_argument("--validation-jsonl")
    p.add_argument("--limit-train",type=int,default=0)
    p.add_argument("--limit-validation",type=int,default=0)
    p.add_argument("--require-quota",action="store_true")
    args=p.parse_args()
    import torch
    torch.set_num_threads(4)
    guard=StorageGuard(ROOT,require_quota=args.require_quota)
    prepare(ROOT/args.out,Codec(device=args.device),guard,args.train_jsonl,args.validation_jsonl,args.limit_train,args.limit_validation)
