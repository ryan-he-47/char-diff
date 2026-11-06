import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.data import LatentCache
from char_diff.model import Denoiser
from char_diff.flow import corrupt,loss_sum,prediction_to_data
from char_diff.utils import ROOT,write_json

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("checkpoint");p.add_argument("--cache")
    p.add_argument("--split",choices=["dev","test"],default="dev")
    p.add_argument("--batches",type=int,default=16);p.add_argument("--device",default="cuda")
    p.add_argument("--precision",choices=["fp32","fp16","bf16"],default="fp32")
    p.add_argument("--weights",choices=["model","ema"],default="ema");p.add_argument("--output",default="runs/evaluation.json")
    args=p.parse_args();torch.set_num_threads(4)
    if args.batches<1:p.error("--batches must be positive")
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    cache=LatentCache(args.cache or ROOT/state["config"]["data"]["cache"],args.split,block=state["config"]["model"]["block"])
    if cache.fingerprint!=state["cache_fingerprint"]:raise ValueError("Evaluation cache differs")
    if len(cache)==0:raise ValueError("Evaluation split has no windows")
    model=Denoiser(state["config"]["model"]).to(args.device).eval();model.load_state_dict(state[args.weights])
    prediction_type=state["config"]["train"].get("prediction_type","x0")
    results=[]
    dtype={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[args.precision]
    with torch.inference_mode(),torch.autocast(device_type=torch.device(args.device).type,dtype=dtype,
                                              enabled=args.device.startswith("cuda") and args.precision!="fp32"):
        for t in [0.05,0.25,0.5,0.75,0.95]:
            total=0.0;count=0;generator=torch.Generator(device=args.device).manual_seed(31415)
            for i in range(min(args.batches,len(cache))):
                batch=cache.batch([i],args.device);x,times=corrupt(batch,generator,t)
                pred=model(x,times,batch["roles"],batch["positions"],batch["lengths"],batch["allowed"])
                data=prediction_to_data(pred,x,times,prediction_type)
                value,n=loss_sum(data,batch);total+=float(value);count+=n
            row={"time":t,"mse":total/(48*count),"units":count};results.append(row);print(row,flush=True)
    write_json(args.output,{"step":state["state"]["step"],"split":args.split,"weights":args.weights,
                           "precision":args.precision,"seed":31415,"cache_fingerprint":cache.fingerprint,
                           "prediction_type":prediction_type,"diffusion_block":cache.block,"cache_packing_block":cache.packing_block,"metric":"x0_mse","buckets":results})
