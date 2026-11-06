"""Exact byte-unit prediction with true block history and pure-noise target blocks."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.model import Denoiser,CLEAN,NOISY,BOS
from char_diff.utils import ROOT,write_json


def target_ids(cache,index):
    parts=cache.segments[int(cache.windows[index]):int(cache.windows[index+1])]
    return np.concatenate([cache.ids[int(cache.offsets[int(doc)])+int(start):int(cache.offsets[int(doc)])+int(end)]
                           for doc,start,end,_ in parts])


def empty_counts(block):
    return {"units":0,"exact":0,"first_units":0,"first_exact":0,
            "offset_units":[0]*block,"offset_exact":[0]*block}


def rates(counts):
    result=dict(counts)
    result["unit_em"]=counts["exact"]/counts["units"]
    result["first_unit_em"]=counts["first_exact"]/counts["first_units"]
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows",type=int,default=64);p.add_argument("--seeds",default="42,123")
    p.add_argument("--steps",type=int,default=32);p.add_argument("--device",default="cuda")
    p.add_argument("--split",choices=["train","dev","test"],default="train")
    p.add_argument("--out",default="runs/teacher_forced_em_20261101")
    p.add_argument("--checkpoint",action="append",help="NAME=PATH; repeat to evaluate explicit checkpoints")
    p.add_argument("--sampler",choices=["euler","gm_ode"],default="euler",help="Sampler for GMM checkpoints; point models use their original Euler step")
    p.add_argument("--substeps",type=int,default=4)
    p.add_argument("--evidence-output",help="Optional explicit evidence destination; existing experiment records stay in their own directory")
    args=p.parse_args();torch.set_num_threads(4)
    if args.windows<1 or args.steps<1 or args.substeps<1:p.error("Positive window, step and substep counts are required")
    seeds=[int(v) for v in args.seeds.split(",")]
    folder=ROOT/args.out;folder.mkdir(parents=True,exist_ok=True)
    if (folder/"cases.jsonl").exists():raise FileExistsError("Choose a new evaluation output directory")
    cache=LatentCache(ROOT/"data/tinystories",args.split);codec=Codec(device=args.device)
    chosen=np.random.default_rng(20261101).choice(len(cache),min(args.windows,len(cache)),replace=False).tolist()
    offsets=np.fromfile(cache.root/"unit_offsets.u64",dtype="<u8");blob=(cache.root/"units.bin").read_bytes()
    raw=lambda ident:blob[int(offsets[ident]):int(offsets[ident+1])]
    mean=torch.tensor(cache.mean,device=args.device);scale=cache.scale
    paths={"x0":"runs/tinystories_x0_22m/last.pt","v":"runs/prediction_target_review_20261101/v/last.pt",
           "v_x0loss":"runs/prediction_target_review_20261101/v_x0loss/last.pt"}
    if args.checkpoint:paths=dict(value.split("=",1) for value in args.checkpoint)
    protocol={"split":args.split,"selection_seed":20261101,"windows":chosen,"seeds":seeds,"steps":args.steps,
              "weights":"ema","precision":"fp32","codec_precision":"fp32","block":cache.block,
              "start":"independent N(0,I), no target-unit input in noisy stream",
              "history":"ground-truth clean history strictly before each block; current/future clean keys masked",
              "score":"exact original bytes, including whitespace, punctuation and case",
              "unique_target_units":cache.target_units(chosen),"cache_fingerprint":cache.fingerprint,
              "torch":torch.__version__,"device":str(args.device),"gmm_sampler":args.sampler,"gmm_substeps":args.substeps}
    write_json(folder/"protocol.json",protocol)
    report={"protocol":protocol,"models":{}};started=time.perf_counter();reference_checked=False
    with open(folder/"cases.jsonl","x",encoding="utf-8",buffering=1) as journal:
        for name,path in paths.items():
            state=torch.load(ROOT/path,map_location="cpu",weights_only=False)
            if state["cache_fingerprint"]!=cache.fingerprint or state["codec"]!=codec.fingerprint:raise ValueError("Input representations differ")
            model=Denoiser(state["config"]["model"]).to(args.device).eval();model.load_state_dict(state["ema"])
            kind=state["config"]["train"].get("prediction_type","x0")
            components=int(model.cfg.get("gmm_components",0))
            actual_sampler=args.sampler if components else "euler"
            step=state["state"]["step"];del state
            generators={s:torch.Generator(device=args.device).manual_seed(s) for s in seeds}
            by_seed={s:empty_counts(cache.block) for s in seeds};examples={"correct":[],"wrong":[]}
            for number,index in enumerate(chosen,1):
                batch=cache.batch([index],args.device);roles=batch["roles"][0]
                ci=torch.nonzero((roles==CLEAN)|(roles==BOS)).flatten();ni=torch.nonzero(roles==NOISY).flatten()
                clean=lambda key:batch[key].index_select(1,ci)
                query=lambda key:batch[key].index_select(1,ni)
                cc=batch["allowed"].index_select(-2,ci).index_select(-1,ci)
                nc=batch["allowed"].index_select(-2,ni).index_select(-1,ci)
                nn=batch["allowed"].index_select(-2,ni).index_select(-1,ni)
                allowed=torch.cat([nc,nn],-1)
                _,past=model(clean("x"),torch.zeros_like(clean("positions"),dtype=torch.float32),clean("roles"),
                             clean("positions"),clean("lengths"),cc,return_cache=True)
                ids=target_ids(cache,index);truth=[raw(int(i)) for i in ids]
                if len(truth)!=len(ni):raise ValueError("Ground-truth target alignment differs")
                if not reference_checked:
                    torch.testing.assert_close(query("x")[0].cpu(),torch.from_numpy(cache.vectors[ids]),atol=0,rtol=0)
                offset=(query("positions")-query("starts"))[0].cpu().numpy()
                for seed in seeds:
                    z=torch.randn(1,len(ni),48,device=args.device,generator=generators[seed])
                    for iteration in range(args.steps):
                        t=1-iteration/args.steps;tn=1-(iteration+1)/args.steps
                        times=torch.full((1,len(ni)),t,device=args.device)
                        pred=model(z,times,query("roles"),query("positions"),query("lengths"),allowed,past=past)
                        if not reference_checked:
                            full=batch["x"].clone();full[:,ni]=z;full_times=torch.zeros_like(batch["positions"],dtype=torch.float32);full_times[:,ni]=t
                            other=model(full,full_times,batch["roles"],batch["positions"],batch["lengths"],batch["allowed"]).index_select(1,ni)
                            error=float((other-pred).abs().max());torch.testing.assert_close(other,pred,atol=5e-5,rtol=5e-5)
                            protocol["cached_vs_full_max_error"]=error;reference_checked=True
                        if components:
                            from char_diff.gmm import mixture_mean,ode_step
                            z=(ode_step(pred,z,t,tn,float(model.cfg["gmm_sigma"]),args.substeps) if actual_sampler=="gm_ode"
                               else (tn/t)*z+(1-tn/t)*mixture_mean(pred))
                        else:
                            z=(tn/t)*z+(1-tn/t)*pred.float() if kind=="x0" else z+(tn-t)*pred.float()
                    if not torch.isfinite(z).all():raise FloatingPointError("Non-finite conditional output")
                    predicted=codec.decode(z[0]*scale+mean)
                    exact=np.array([a==b for a,b in zip(predicted,truth)],dtype=bool)
                    if len(predicted)!=len(truth):raise ValueError("Decoder target count differs")
                    first=offset==0;counts=by_seed[seed]
                    counts["units"]+=len(truth);counts["exact"]+=int(exact.sum());counts["first_units"]+=int(first.sum());counts["first_exact"]+=int(exact[first].sum())
                    for i in range(cache.block):
                        mask=offset==i;counts["offset_units"][i]+=int(mask.sum());counts["offset_exact"][i]+=int(exact[mask].sum())
                    row={"model":name,"window":index,"seed":seed,"units":len(truth),"exact":int(exact.sum()),"first_units":int(first.sum()),"first_exact":int(exact[first].sum())}
                    journal.write(json.dumps(row)+"\n")
                    for i,(a,b) in enumerate(zip(predicted,truth)):
                        label="correct" if a==b else "wrong"
                        if len(examples[label])<32:
                            examples[label].append({"window":index,"seed":seed,"unit_id":int(ids[i]),"offset":int(offset[i]),
                                                    "target_hex":b.hex(),"predicted_hex":a.hex(),"target":b.decode("utf-8",errors="replace"),"predicted":a.decode("utf-8",errors="replace")})
                if number%8==0:
                    print(json.dumps({"model":name,"windows":number,"by_seed":{s:{"unit_em":rates(c)["unit_em"],"first_unit_em":rates(c)["first_unit_em"]} for s,c in by_seed.items()}}),flush=True)
            total=empty_counts(cache.block)
            for c in by_seed.values():
                for key in total:
                    if isinstance(total[key],list):total[key]=[a+b for a,b in zip(total[key],c[key])]
                    else:total[key]+=c[key]
            report["models"][name]={"checkpoint":path,"step":step,"prediction_type":kind,"sampler":actual_sampler,"gmm_components":components,
                                   "aggregate":rates(total),"by_seed":{s:rates(c) for s,c in by_seed.items()},"examples":examples}
            write_json(folder/"progress.json",report)
            print(json.dumps({"model":name,"aggregate":rates(total)}),flush=True)
            del model,past
    report["elapsed_seconds"]=time.perf_counter()-started
    write_json(folder/"summary.json",report)
    if args.evidence_output:write_json(ROOT/args.evidence_output,report)
    print(json.dumps({name:r["aggregate"] for name,r in report["models"].items()}),flush=True)


if __name__=="__main__":main()
