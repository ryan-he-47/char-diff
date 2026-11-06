from __future__ import annotations
import torch
from .model import CLEAN,NOISY,BOS

def corrupt(batch,generator=None,fixed_t=None,time_sampling="uniform"):
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
    return noisy,t

def loss_sum(prediction,batch):
    valid=batch["roles"]==NOISY
    error=(prediction.float()-batch["x"].float()).square().sum(-1)
    return error[valid].sum(),int(valid.sum().item())

@torch.inference_mode()
def generate(model,prefix,units,steps=32,seed=42,use_cache=True,history_transform=None):
    """Generate raw standardized latents; an optional transform changes future history only."""
    if units < 0 or steps < 1:
        raise ValueError("Generation requires units >= 0 and steps >= 1")
    device=next(model.parameters()).device
    model.eval()
    prefix=torch.as_tensor(prefix,dtype=torch.float32,device=device).reshape(-1,model.cfg.get("latent_dim",48))
    history=prefix.clone()
    generator=torch.Generator(device=device).manual_seed(seed)
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
        length=min(model.cfg["block"],units-sum(len(z) for z in produced))
        z=torch.randn(1,length,48,device=device,generator=generator)
        start=len(history)+1
        pos=torch.arange(start,start+length,device=device)[None]
        roles=torch.full((1,length),NOISY,device=device,dtype=torch.long)
        lengths=torch.full_like(roles,length)
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
            z=(tn/t)*z+(1-tn/t)*estimate.float()
        produced.append(z[0])
        clean_block=z[0]
        if history_transform is not None:
            # A callback may mutate its input; protect the current block's emitted values.
            clean_block=history_transform(z[0].clone())
            if not isinstance(clean_block,torch.Tensor) or clean_block.shape!=z[0].shape:
                raise ValueError("History transform must preserve the block shape")
            if clean_block.device!=device or not torch.isfinite(clean_block).all():
                raise ValueError("History transform must return finite latents on the model device")
            clean_block=clean_block.to(dtype=z.dtype)
        history=torch.cat([history,clean_block])
        if use_cache:
            allowed=torch.cat([torch.ones(length,start,device=device,dtype=torch.bool),torch.ones(length,length,device=device,dtype=torch.bool).tril()],1)[None,None]
            _,cache=model(clean_block[None],torch.zeros(1,length,device=device),torch.full_like(roles,CLEAN),pos,
                          torch.zeros_like(lengths),allowed,past=cache,return_cache=True)
    return torch.cat(produced) if produced else torch.empty(0,48,device=device)
