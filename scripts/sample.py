import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec,HistoryReencoder
from char_diff.model import Denoiser
from char_diff.flow import generate
from char_diff.selection import CycleScorer,CycleRetry
from char_diff.utils import write_json

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint");p.add_argument("--prompt",default="Once upon a time, there was a little girl named Lily. ")
    p.add_argument("--units",type=int,default=128);p.add_argument("--steps",type=int,default=32)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--weights",choices=["model","ema"],default="ema")
    p.add_argument("--output",default="runs/sample.json");p.add_argument("--device",default="cuda")
    p.add_argument("--precision",choices=["fp32","fp16","bf16"],default="fp32")
    p.add_argument("--history",choices=["raw","reencode"],default="raw")
    p.add_argument("--sampler",choices=["euler","gm_ode"],default="euler")
    p.add_argument("--substeps",type=int,default=4)
    p.add_argument("--commit-first",action="store_true",help="Denoise a full block, commit only its first unit, and repeat")
    p.add_argument("--cycle-retry-threshold",type=float,default=None,help="B1 only: retry once above this AE cycle RMSE")
    args=p.parse_args();torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    codec=Codec(device=args.device)
    if state["codec"]!=codec.fingerprint:raise ValueError("Checkpoint codec differs")
    model=Denoiser(state["config"]["model"]).to(args.device).eval();model.load_state_dict(state[args.weights])
    mean=np.asarray(state["normalizer"]["mean"],np.float32);scale=state["normalizer"]["scale"]
    units=codec.split([args.prompt])[0] if args.prompt else []
    prefix=(codec.encode(units)-mean)/scale
    transform=HistoryReencoder(codec,state["normalizer"]) if args.history=="reencode" else None
    selector=CycleRetry(CycleScorer(codec,state["normalizer"]),args.cycle_retry_threshold) if args.cycle_retry_threshold is not None else None
    import time
    if args.device.startswith("cuda"):torch.cuda.synchronize()
    started=time.perf_counter()
    dtype={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[args.precision]
    with torch.autocast(device_type=torch.device(args.device).type,dtype=dtype,
                        enabled=args.device.startswith("cuda") and args.precision!="fp32"):
        z=generate(model,prefix,args.units,args.steps,args.seed,history_transform=transform,
                   prediction_type=state["config"]["train"].get("prediction_type","x0"),
                   sampler=args.sampler,substeps=args.substeps,commit_first=args.commit_first,unit_selector=selector)
    if not torch.isfinite(z).all():raise FloatingPointError("Non-finite generated latents")
    raw=codec.decode(z*scale+torch.as_tensor(mean,device=z.device))
    generated=b"".join(raw)
    try:generated.decode("utf-8");utf8=True
    except UnicodeDecodeError:utf8=False
    result={"prompt":args.prompt,"generated":generated.decode("utf-8",errors="replace"),
            "bytes_hex":generated.hex(),"unit_hex":[u.hex() for u in raw],"utf8_valid":utf8,
            "units":args.units,"steps_per_block":args.steps,"denoiser_calls":((args.units if args.commit_first else (args.units+model.cfg["block"]-1)//model.cfg["block"])+(selector.summary()['retries'] if selector else 0))*args.steps,
            "commit_first":args.commit_first,
            "seed":args.seed,"weights":args.weights,"precision":args.precision,"training_step":state["state"]["step"],
            "checkpoint":str(Path(args.checkpoint).resolve()),"cache_fingerprint":state["cache_fingerprint"],
            "history":args.history,"history_stats":transform.summary() if transform else None,
            "selection":selector.summary() if selector else None,"selection_trace":selector.records if selector else None,
            "sampler":args.sampler,"substeps":args.substeps if args.sampler=="gm_ode" else 1,
            "prediction_type":state["config"]["train"].get("prediction_type","x0"),
            "seconds":time.perf_counter()-started}
    write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=True),flush=True)
