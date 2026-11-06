from __future__ import annotations
import torch
from .model import CLEAN,NOISY,BOS

def corrupt(batch,generator=None,fixed_t=None,time_sampling="uniform",return_noise=False):
    target=batch["x"]
    indices=batch["noise_block"]
    n=int(indices.max().item())+1
    if time_sampling not in ("uniform","beta_2_1"):
        raise ValueError("Choose uniform or beta_2_1 time sampling")
    times=torch.rand(n,device=target.device,generator=generator) if fixed_t is None else torch.full((n,),float(fixed_t),device=target.device)
    if fixed_t is None and time_sampling=="beta_2_1":
        # Inverse CDF of Beta(2,1); paired arms consume identical random draws.
        times=times.sqrt()
    t=torch.where(indices>=0,times[indices.clamp_min(0)],0.0)
    eps=torch.randn(target.shape,device=target.device,generator=generator,dtype=target.dtype)
    noisy=(1-t[...,None])*target+t[...,None]*eps
    return (noisy,t,eps) if return_noise else (noisy,t)

def loss_sum(prediction,batch,target=None,weights=None):
    valid=batch["roles"]==NOISY
    target=batch["x"] if target is None else target
    error=(prediction.float()-target.float()).square().sum(-1)
    if weights is not None:error=error*weights.float()
    return error[valid].sum(),int(valid.sum().item())


def prediction_to_data(prediction,noisy,times,prediction_type="x0"):
    if prediction_type=="x0":
        if prediction.shape[-1] != noisy.shape[-1]:
            from .gmm import mixture_mean
            return mixture_mean(prediction,noisy.shape[-1])
        return prediction
    if prediction_type=="v":return noisy.float()-times[...,None]*prediction.float()
    raise ValueError("Choose x0 or v prediction")

@torch.inference_mode()
def generate(model,prefix,units,steps=32,seed=42,use_cache=True,history_transform=None,prediction_type="x0",sampler="euler",substeps=4,commit_first=False,unit_selector=None,block_schedule=None,stop_condition=None,max_blocks=None):
    """Generate latents; commit_first denoises full blocks and retains their first unit."""
    if units < 0 or steps < 1:
        raise ValueError("Generation requires units >= 0 and steps >= 1")
    if prediction_type not in ("x0","v"):raise ValueError("Choose x0 or v prediction")
    components=int(model.cfg.get("gmm_components",0))
    if sampler not in ("euler","gm_ode") or substeps<1:raise ValueError("Unknown sampler or invalid substeps")
    if components and prediction_type!="x0":raise ValueError("This GMM predicts x0")
    if sampler=="gm_ode" and not components:raise ValueError("GM-ODE requires a GMM checkpoint")
    if unit_selector is not None and (model.cfg["block"]!=1 or components or prediction_type!="x0" or sampler!="euler"):
        raise ValueError("Unit selection currently requires B1 point x0 / Euler")
    device=next(model.parameters()).device
    model.eval()
    prefix=torch.as_tensor(prefix,dtype=torch.float32,device=device).reshape(-1,model.cfg.get("latent_dim",48))
    history=prefix.clone()
    generator=torch.Generator(device=device).manual_seed(seed)
    retry_generator=torch.Generator(device=device).manual_seed(seed+1000003) if unit_selector is not None else None
    def prefill(data):
        z=torch.cat([torch.zeros_like(data[:1]) if len(data) else torch.zeros(1,48,device=device),data])[None]
        n=z.shape[1]
        roles=torch.full((1,n),CLEAN,device=device,dtype=torch.long);roles[:,0]=BOS
        positions=torch.arange(n,device=device)[None]
        lengths=torch.zeros_like(roles)
        _,cache=model(z,torch.zeros(1,n,device=device),roles,positions,lengths,
                      torch.ones(n,n,device=device,dtype=torch.bool).tril()[None,None],return_cache=True)
        return cache
    cache=prefill(history) if use_cache else None
    produced=[]
    while sum(len(z) for z in produced)<units:
        if max_blocks is not None and len(produced)>=max_blocks:break
        requested=int(block_schedule(produced)) if block_schedule is not None else model.cfg["block"]
        if not 1<=requested<=model.cfg.get('length_capacity',model.cfg['block']):
            raise ValueError('Requested block exceeds the trained length capacity')
        if unit_selector is not None and requested!=1:raise ValueError('Unit selector requires length one')
        length=requested if commit_first else min(requested,units-sum(len(z) for z in produced))
        z=torch.randn(1,length,48,device=device,generator=generator)
        start=len(history)+1
        pos=torch.arange(start,start+length,device=device)[None]
        roles=torch.full((1,length),NOISY,device=device,dtype=torch.long)
        lengths=torch.full_like(roles,length)
        def denoise(z):
            for step in range(steps):
                t=1-step/steps;tn=1-(step+1)/steps
                if use_cache:
                    allow=torch.ones(1,1,length,start+length,device=device,dtype=torch.bool)
                    estimate=model(z,torch.full((1,length),t,device=device),roles,pos,lengths,allow,past=cache)
                else:
                    clean=torch.cat([torch.zeros(1,48,device=device),history])[None]
                    all_x=torch.cat([clean,z],1);nc=clean.shape[1];n=nc+length
                    all_roles=torch.cat([torch.full((1,nc),CLEAN,device=device,dtype=torch.long),roles],1);all_roles[:,0]=BOS
                    all_pos=torch.cat([torch.arange(nc,device=device)[None],pos],1)
                    all_len=torch.cat([torch.zeros(1,nc,device=device,dtype=torch.long),lengths],1)
                    time=torch.cat([torch.zeros(1,nc,device=device),torch.full((1,length),t,device=device)],1)
                    allow=torch.zeros(1,1,n,n,device=device,dtype=torch.bool)
                    allow[0,0,:nc,:nc]=torch.ones(nc,nc,device=device,dtype=torch.bool).tril()
                    allow[0,0,nc:,:]=True
                    estimate=model(all_x,time,all_roles,all_pos,all_len,allow)[:,nc:]
                if components:
                    from .gmm import mixture_mean,ode_step
                    if sampler=="gm_ode":
                        z=ode_step(estimate,z,t,tn,float(model.cfg["gmm_sigma"]),substeps)
                    else:
                        z=(tn/t)*z+(1-tn/t)*mixture_mean(estimate)
                elif prediction_type=="x0":
                    z=(tn/t)*z+(1-tn/t)*estimate.float()
                else:
                    # t=0 is data, t=1 is noise; v=eps-x and dt is negative.
                    z=z+(tn-t)*estimate.float()
            return z
        z=denoise(z)
        committed=1 if commit_first else length
        clean_block=z[0,:committed]
        if unit_selector is not None:
            # Both candidates see exactly the same history/KV; commit only once.
            def retry():
                noise=torch.randn(1,length,48,device=device,generator=retry_generator)
                return denoise(noise)[0]
            clean_block=unit_selector(clean_block.clone(),retry)
            if clean_block.shape!=(1,48) or clean_block.device!=device or not torch.isfinite(clean_block).all():
                raise ValueError("Unit selector must return one finite latent on the model device")
        produced.append(clean_block)
        if history_transform is not None:
            # A callback may mutate its input; protect the current block's emitted values.
            clean_block=history_transform(clean_block.clone())
            if not isinstance(clean_block,torch.Tensor) or clean_block.shape!=produced[-1].shape:
                raise ValueError("History transform must preserve the block shape")
            if clean_block.device!=device or not torch.isfinite(clean_block).all():
                raise ValueError("History transform must return finite latents on the model device")
            clean_block=clean_block.to(dtype=z.dtype)
        history=torch.cat([history,clean_block])
        if use_cache:
            allowed=torch.cat([torch.ones(committed,start,device=device,dtype=torch.bool),torch.ones(committed,committed,device=device,dtype=torch.bool).tril()],1)[None,None]
            _,cache=model(clean_block[None],torch.zeros(1,committed,device=device),torch.full_like(roles[:,:committed],CLEAN),pos[:,:committed],
                          torch.zeros_like(lengths[:,:committed]),allowed,past=cache,return_cache=True)
        if stop_condition is not None and stop_condition(produced):break
    return torch.cat(produced) if produced else torch.empty(0,48,device=device)
