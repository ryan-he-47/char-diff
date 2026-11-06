from __future__ import annotations
import copy
import json
import math
import os
import random
import signal
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from .data import LatentCache,WindowSampler
from .flow import corrupt,loss_sum
from .model import Denoiser
from .storage import StorageGuard
from .utils import ROOT,write_json,sha256

def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(s):
    random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["torch"])
    if s["cuda"]:torch.cuda.set_rng_state_all(s["cuda"])

def learning_rate(seen,budget,cfg):
    warm=max(1,budget*cfg["warmup_fraction"]);decay=budget*(1-cfg["decay_fraction"])
    if seen<warm:return cfg["lr"]*max(seen,1)/warm
    if seen>=decay:return cfg["lr"]*max(0,(budget-seen)/max(1,budget-decay))
    return cfg["lr"]

def atomic_checkpoint(path,payload,guard,estimate,rotate=False):
    path=Path(path);guard.check(estimate);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".pending.pt")
    with open(tmp,"wb") as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    if rotate and path.exists():os.replace(path,path.with_name("previous.pt"))
    os.replace(tmp,path)

def train(cfg,resume=None,stop_after=0,device=None,fork_from=None,gmm_from=None,block_from=None):
    if sum(bool(v) for v in (resume,fork_from,gmm_from,block_from))>1:
        raise ValueError("Choose exact resume, continuation, GMM warm-start, or block-size warm-start")
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(4)
    tc=cfg["train"];seed=int(tc["seed"])
    autoregressive=cfg["model"].get("family")=="gmm_ar"
    time_sampling=tc.get("time_sampling","uniform")
    prediction_type=tc.get("prediction_type","x0");loss_space=tc.get("loss_space","x0")
    if time_sampling not in ("uniform","beta_2_1"):raise ValueError("Unknown time sampling")
    if autoregressive:
        if prediction_type!="gmm_ar" or loss_space!="nll":raise ValueError("AR GMM requires the raw-AE NLL objective")
        if fork_from or gmm_from or block_from:raise ValueError("AR GMM starts fresh or resumes its own full state")
    elif prediction_type not in ("x0","v") or loss_space not in ("x0","native"):
        raise ValueError("Unknown prediction or loss space")
    components=int(cfg["model"].get("gmm_components",0))
    if components and (prediction_type!="x0" or loss_space!="x0"):
        raise ValueError("Fixed-width GMM uses the scaled x0 likelihood objective")
    if components and not (gmm_from or resume or fork_from):
        raise ValueError("Initialize a GMM explicitly with --gmm-from or resume its full state")
    if gmm_from and not components:raise ValueError("GMM warm-start requires a GMM output head")
    if block_from and (components or prediction_type!="x0" or loss_space!="x0"):
        raise ValueError("Block-size warm-start retains the point x0 objective")
    for key in ("batch_windows", "micro_batch", "log_every", "save_every"):
        if int(tc[key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    if tc["epochs"] <= 0 or tc.get("max_steps", 0) < 0 or stop_after < 0:
        raise ValueError("Require positive epochs and nonnegative step limits")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    guard=StorageGuard(ROOT,**dict(limit_gib=cfg["storage"]["project_limit_gib"],reserve_gib=cfg["storage"]["reserve_gib"],require_quota=cfg["storage"].get("require_quota",False)))
    guard.check(512*1024**2)
    if autoregressive:
        from .ar import ARCache,GMMAR
    cache_type=ARCache if autoregressive else LatentCache
    cache=cache_type(ROOT/cfg["data"]["cache"],block=cfg["data"]["block"])
    if any(cache.manifest["layout"][k]!=cfg["data"][k] for k in ("window","carry")):
        raise ValueError("Cache/data window or carry mismatch")
    if not autoregressive and cache.block!=cfg["model"]["block"]:raise ValueError("Cache/model block size mismatch")
    if cfg["model"].get("latent_dim",48)!=48:raise ValueError("This codec uses 48-dimensional latents")
    model=(GMMAR if autoregressive else Denoiser)(cfg["model"]).to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    latent_mean=torch.tensor(cache.mean,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc["lr"],betas=tuple(tc["betas"]),eps=tc["eps"],weight_decay=tc["weight_decay"])
    dtype={"bf16":torch.bfloat16,"fp16":torch.float16,"fp32":torch.float32}[tc["precision"]]
    amp=device.type=="cuda" and dtype!=torch.float32
    if amp and dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():raise ValueError("Choose fp16 for this GPU")
    scaler=torch.amp.GradScaler("cuda",enabled=amp and dtype==torch.float16,init_scale=1024,growth_interval=500)
    sampler=WindowSampler(len(cache),seed)
    budget=int(cache.units*tc["epochs"])
    if tc.get("max_steps",0):budget=int(tc["max_steps"]*tc["batch_windows"]*cache.units/len(cache))
    reference=None;reference_sha=None;limit_steps=int(tc.get("max_steps",0))
    if tc.get("reference_schedule"):
        if fork_from or limit_steps:raise ValueError("Reference schedule uses global updates; choose weight warm-start or exact resume")
        reference_path=ROOT/tc["reference_schedule"]
        reference=json.loads(reference_path.read_text(encoding="utf-8"));reference_sha=sha256(reference_path)
        limit_steps=int(reference["total_steps"]);budget=int(reference["total_seen_units"])
        rates=reference["learning_rates"]
        if reference["epoch_units"]!=cache.units or len(rates)!=limit_steps or limit_steps<1 or budget<1:
            raise ValueError("Reference schedule does not match the training corpus or update count")
        if any(not math.isfinite(v) or not 0<=v<=tc["lr"] for v in rates):
            raise ValueError("Invalid reference learning rates")
    state={"step":0,"seen_units":0,"attempted_units":0,"skipped_units":0,"attempts":0,"saved_milestones":[]}
    norms=deque(maxlen=100);run=Path(ROOT/tc["run_dir"]);run.resolve().relative_to(ROOT.resolve())
    if run.exists() and any((run/name).exists() for name in ("last.pt","train.jsonl","config.json")) and not resume:
        raise FileExistsError("Run already contains training artifacts; choose a new directory or exact resume")
    run.mkdir(parents=True,exist_ok=True)
    lineage=None
    if resume or fork_from:
        loaded=torch.load(resume or fork_from,map_location="cpu",weights_only=False)
        if loaded["cache_fingerprint"]!=cache.fingerprint:raise ValueError("Resume cache differs")
        if loaded["config"]["model"]!=cfg["model"]:raise ValueError("Resume model differs")
        if resume and loaded["budget_units"]!=budget:raise ValueError("Resume training budget differs")
        # Exact continuation preserves the objective, schedule and random draw layout.
        # Logging/storage paths can change without changing the optimization run.
        keys=("seed", "epochs", "max_steps", "batch_windows", "micro_batch", "precision",
                    "lr", "betas", "eps", "weight_decay", "clip", "ema", "warmup_fraction", "decay_fraction",
                    "reference_schedule", "gmm_init_rms", "gmm_init_seed", "gmm_source_sha256", "block_source_sha256")
        for key in keys:
            if fork_from and key in ("epochs","max_steps","lr","warmup_fraction","decay_fraction"):continue
            if loaded["config"]["train"].get(key) != tc.get(key):
                raise ValueError(f"Resume train.{key} differs; use the original configuration")
        if resume and loaded["config"]["train"].get("time_sampling","uniform")!=time_sampling:
            raise ValueError("Resume train.time_sampling differs")
        if resume and loaded.get("reference_schedule_sha256")!=reference_sha:
            raise ValueError("Resume reference learning-rate schedule differs")
        for key,default in (("prediction_type","x0"),("loss_space","x0")):
            if loaded["config"]["train"].get(key,default)!=tc.get(key,default):
                raise ValueError(f"Resume/fork train.{key} differs; start a new model for a new target")
        if "optimizer" not in loaded:
            raise ValueError("Exact resume requires last.pt or previous.pt with optimizer state")
        model.load_state_dict(loaded["model"]);ema.load_state_dict(loaded["ema"])
        optimizer.load_state_dict(loaded["optimizer"]);scaler.load_state_dict(loaded["scaler"])
        sampler.load_state_dict(loaded["sampler"]);norms.extend(loaded["norms"])
        if resume:
            state=loaded["state"];lineage=loaded.get("lineage")
        else:
            if Path(fork_from).resolve().parent==run.resolve():raise ValueError("A continuation requires a separate run directory")
            lineage={"kind":"continuation","source_sha256":sha256(fork_from),"source_step":loaded["state"]["step"],
                     "source_seen_units":loaded["state"]["seen_units"],"source_checkpoint":str(fork_from),
                     "source_time_sampling":loaded["config"]["train"].get("time_sampling","uniform"),
                     "restored":["model","ema","optimizer","scaler","sampler","rng","norms"],
                     "schedule":"new branch budget; counters reset; WSD restarted"}
        restore_rng(loaded["rng"])
    elif block_from:
        if reference is None:raise ValueError("Block-size experiment requires the shared reference schedule")
        loaded=torch.load(block_from,map_location="cpu",weights_only=False)
        digest=sha256(block_from)
        if tc.get("block_source_sha256") and tc["block_source_sha256"]!=digest:
            raise ValueError("Block-size source checkpoint differs")
        if Path(block_from).resolve().parent==run.resolve():raise ValueError("Warm-start requires a separate run directory")
        if loaded["cache_fingerprint"]!=cache.fingerprint:raise ValueError("Warm-start cache differs")
        source_model=copy.deepcopy(loaded["config"]["model"])
        source_block=source_model["block"]
        source_model.setdefault("length_capacity",source_block)
        candidate=copy.deepcopy(cfg["model"]);candidate.setdefault("length_capacity",candidate["block"])
        candidate["block"]=source_block
        if candidate!=source_model:raise ValueError("Block-size warm-start must preserve the full backbone and length table")
        source_train=loaded["config"]["train"]
        if source_train.get("prediction_type","x0")!="x0" or source_model.get("gmm_components",0):
            raise ValueError("Block-size source must use point x0 prediction")
        for key in ("batch_windows","micro_batch","precision","seed","lr","betas","eps","weight_decay","clip","ema"):
            if source_train[key]!=tc[key]:raise ValueError(f"Block-size warm-start train.{key} differs")
        model.load_state_dict(loaded["model"]);ema.load_state_dict(loaded["ema"])
        state=copy.deepcopy(loaded["state"])
        if not 0<=state["step"]<limit_steps:raise ValueError("Warm-start must precede the shared endpoint")
        consumed=int(state["attempts"])*int(tc["batch_windows"])
        cursor={"size":len(cache),"seed":seed,"epoch":consumed//len(cache),"cursor":consumed%len(cache)}
        if "sampler" in loaded and loaded["sampler"]!=cursor:raise ValueError("Derived source cursor differs")
        sampler.load_state_dict(cursor)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device.type=="cuda":torch.cuda.manual_seed_all(seed)
        lineage={"kind":"block_size_weight_warm_start","source_checkpoint":str(block_from),
                 "source_sha256":digest,"source_step":state["step"],"source_seen_units":state["seen_units"],
                 "source_block":source_block,"target_block":cache.block,"source_has_optimizer":"optimizer" in loaded,
                 "restored":["model","ema","data_cursor","global_counters","original_length_embedding_table"],
                 "reinitialized":["optimizer","scaler","gradient_norm_window","rng"],
                 "schedule":"global successful-update learning rates from the existing x0 control",
                 "reference_schedule_sha256":reference_sha,"target_step":limit_steps}
        del loaded
    elif gmm_from:
        from .gmm import expand_output_state
        if reference is None:raise ValueError("GMM timing experiment requires the shared reference schedule")
        loaded=torch.load(gmm_from,map_location="cpu",weights_only=False)
        source_digest=sha256(gmm_from)
        if tc.get("gmm_source_sha256") and tc["gmm_source_sha256"]!=source_digest:
            raise ValueError("GMM source checkpoint differs from the approved branch source")
        if Path(gmm_from).resolve().parent==run.resolve():raise ValueError("Warm-start requires a separate run directory")
        if loaded["cache_fingerprint"]!=cache.fingerprint:raise ValueError("Warm-start cache differs")
        base_model={k:v for k,v in cfg["model"].items() if k not in ("gmm_components","gmm_sigma")}
        if loaded["config"]["model"]!=base_model:raise ValueError("Warm-start backbone differs")
        source_train=loaded["config"]["train"]
        if source_train.get("prediction_type","x0")!="x0":raise ValueError("Warm-start source must predict x0")
        for key in ("batch_windows","micro_batch","precision","seed"):
            if source_train[key]!=tc[key]:raise ValueError(f"Warm-start train.{key} differs")
        for name,network in (("model",model),("ema",ema)):
            expanded=expand_output_state(loaded[name],components,float(tc["gmm_init_rms"]),int(tc["gmm_init_seed"]))
            network.load_state_dict(expanded)
        state=copy.deepcopy(loaded["state"])
        if not 0<=state["step"]<limit_steps:raise ValueError("Warm-start must precede the shared endpoint")
        consumed=int(state["attempts"])*int(tc["batch_windows"])
        cursor={"size":len(cache),"seed":seed,"epoch":consumed//len(cache),"cursor":consumed%len(cache)}
        if "sampler" in loaded and loaded["sampler"]!=cursor:raise ValueError("Derived source cursor differs")
        sampler.load_state_dict(cursor)
        # Weight-only milestones and full endpoints use the same intervention.
        # Head initialization has its own generator; noise starts from this seed.
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device.type=="cuda":torch.cuda.manual_seed_all(seed)
        lineage={"kind":"gmm_weight_warm_start","source_checkpoint":str(gmm_from),
                 "source_sha256":source_digest,"source_step":state["step"],
                 "source_seen_units":state["seen_units"],"source_has_optimizer":"optimizer" in loaded,
                 "restored":["model_backbone","ema_backbone","data_cursor","global_counters"],
                 "reinitialized":["optimizer","scaler","gradient_norm_window","rng"],
                 "head":{"components":components,"sigma":cfg["model"]["gmm_sigma"],
                         "init_rms":tc["gmm_init_rms"],"init_seed":tc["gmm_init_seed"]},
                 "schedule":"global successful-update learning rates from the existing x0 control",
                 "reference_schedule_sha256":reference_sha,"target_step":limit_steps}
        del loaded,expanded
    parameters=sum(p.numel() for p in model.parameters())
    source_root=Path(__file__).resolve().parents[1]
    source_sha256={name:sha256(source_root/name) for name in ("char_diff/training.py","char_diff/flow.py","char_diff/gmm.py","char_diff/model.py","char_diff/data.py","scripts/train.py")}
    if autoregressive:
        source_sha256.update({name:sha256(source_root/name) for name in ("char_diff/ar.py","char_diff/vendor/bltz/models/mdn.py")})
    write_json(run/"config.json",cfg)
    if lineage:write_json(run/"lineage.json",lineage)
    stop={"requested":False}
    def request_stop(*_):stop["requested"]=True
    old_handlers={sig: signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    start_step=state["step"];start_units=state["seen_units"];started=time.monotonic();last_log=started
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    def payload(full=True):
        p={"format":1,"model":model.state_dict(),"ema":ema.state_dict(),"config":cfg,
           "cache_fingerprint":cache.fingerprint,"codec":cache.manifest["codec"],
           "normalizer":cache.manifest["normalizer"],"state":dict(state),"budget_units":budget,"lineage":lineage,"source_sha256":source_sha256,
           "reference_schedule_sha256":reference_sha}
        if full:p.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),sampler=sampler.state_dict(),rng=rng_state(),norms=list(norms))
        return p
    last_saved_attempt=-1
    def save():
        nonlocal last_saved_attempt
        if state["attempts"]==last_saved_attempt:return
        atomic_checkpoint(run/"last.pt",payload(),guard,int(parameters*18+8*1024**2),rotate=True)
        last_saved_attempt=state["attempts"]
    print(json.dumps({"event":"start","parameters":parameters,"budget_units":budget,"epoch_units":cache.units,"windows":len(cache),"device":str(device),"step":state["step"],"time_sampling":time_sampling,"prediction_type":prediction_type,"loss_space":loss_space,"diffusion_block":cache.block,"cache_packing_block":cache.packing_block,"lineage":lineage}),flush=True)
    log=open(run/"train.jsonl","a",encoding="utf-8",buffering=1)
    consecutive_invalid=0
    try:
        while (state["step"]<limit_steps if limit_steps else state["seen_units"]<budget):
            tick=time.monotonic();indices=sampler.next(tc["batch_windows"]);count=cache.target_units(indices)
            lr=reference["learning_rates"][state["step"]] if reference else learning_rate(state["seen_units"]+count,budget,tc)
            for group in optimizer.param_groups:group["lr"]=lr
            optimizer.zero_grad(set_to_none=True);total_loss=0.0;physical=0;data_seconds=0.0;gmm_stats={}
            model.train()
            for i in range(0,len(indices),tc["micro_batch"]):
                data_tick=time.monotonic()
                batch=cache.batch(indices[i:i+tc["micro_batch"]],device)
                data_seconds+=time.monotonic()-data_tick
                if autoregressive:
                    x=t=target=weights=None
                elif prediction_type=="v":
                    x,t,eps=corrupt(batch,time_sampling=time_sampling,return_noise=True)
                    target=eps-batch["x"]
                    weights=t.square() if loss_space=="x0" else None
                else:
                    x,t=corrupt(batch,time_sampling=time_sampling);target=None;weights=None
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                    if autoregressive:
                        raw,n,ar_stats=model.loss_sum(batch,latent_mean,cache.scale,
                            diagnostics=i==0 and (state["step"]+1)%tc["log_every"]==0)
                        gmm_stats.update(ar_stats)
                    else:
                        prediction=model(x,t,batch["roles"],batch["positions"],batch["lengths"],batch["allowed"])
                    if autoregressive:
                        pass
                    elif components:
                        from .gmm import loss_sum as gmm_loss_sum,diagnostics
                        raw,n=gmm_loss_sum(prediction,batch,float(cfg["model"]["gmm_sigma"]))
                        if i==0 and (state["step"]+1)%tc["log_every"]==0:
                            gmm_stats=diagnostics(prediction,batch,float(cfg["model"]["gmm_sigma"]))
                    else:
                        raw,n=loss_sum(prediction,batch,target=target,weights=weights)
                    denominator=count if autoregressive else 48*count
                    loss=raw/denominator
                scaler.scale(loss).backward();total_loss+=float(raw.detach())/denominator
                physical+=batch["x"].shape[0]*batch["x"].shape[1]
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc["clip"])
            threshold=max(2000,10*float(np.median(norms))) if norms else 2000
            good=bool(torch.isfinite(norm)) and float(norm)<=threshold and math.isfinite(total_loss)
            state["attempts"]+=1;state["attempted_units"]+=count
            if good:
                consecutive_invalid=0
                scaler.step(optimizer);scaler.update();state["step"]+=1;state["seen_units"]+=count;norms.append(float(norm))
                with torch.no_grad():
                    for target,current in zip(ema.parameters(),model.parameters()):target.lerp_(current.detach(),1-tc["ema"])
            else:
                scaler.update();state["skipped_units"]+=count
                consecutive_invalid+=1
            if device.type=="cuda":torch.cuda.synchronize(device)
            row={"step":state["step"],"attempt":state["attempts"],"loss":total_loss,"grad_norm":float(norm),"lr":lr,"time_sampling":time_sampling,"prediction_type":prediction_type,"loss_space":loss_space,
                 "diffusion_block":cache.block,"seen_units":state["seen_units"],"epochs":state["seen_units"]/cache.units,"skipped_units":state["skipped_units"],
                 "attempted_units":state["attempted_units"],"sampler_epoch":sampler.epoch,"sampler_cursor":sampler.cursor,
                 "seconds":time.monotonic()-tick,"target_units":count,"physical_positions":physical,"updated":good,
                 "batch_build_host_seconds":data_seconds,"physical_positions_per_second":physical/max(1e-9,time.monotonic()-tick),
                 "target_units_per_second":(state["seen_units"]-start_units)/max(1e-9,time.monotonic()-started),
                 "peak_cuda_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0}
            row.update(gmm_stats)
            log.write(json.dumps(row)+"\n")
            if state["step"]%tc["log_every"]==0 or not good or time.monotonic()-last_log>30:
                print(json.dumps(row),flush=True);last_log=time.monotonic()
            if consecutive_invalid>=20:
                save()
                raise RuntimeError("20 consecutive invalid updates; recovery checkpoint saved")
            if good:
                for milestone in tc["milestones"]:
                    if state["seen_units"]/cache.units>=milestone and milestone not in state["saved_milestones"] and not tc.get("max_steps",0):
                        state["saved_milestones"].append(milestone)
                        atomic_checkpoint(run/f"weights_epoch_{milestone:g}.pt",payload(False),guard,int(parameters*9+4*1024**2))
                if state["step"]%tc["save_every"]==0:save()
            if stop["requested"] or (stop_after and state["step"]-start_step>=stop_after):break
        save()
    finally:
        log.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    result={"step":state["step"],"seen_units":state["seen_units"],"elapsed_seconds":time.monotonic()-started,
            "checkpoint":str(run/"last.pt"),"stopped_by_request":stop["requested"]}
    write_json(run/"result.json",result);print(json.dumps(result),flush=True)
    return result
