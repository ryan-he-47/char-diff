from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

CLEAN, NOISY, BOS, PAD = range(4)

class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
    def forward(self, x):
        v = x.float()
        return (v * torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)).to(x.dtype) * self.weight.to(x.dtype)

def timestep_embedding(t, dim=256):
    frequencies = torch.exp(-math.log(10000) * torch.arange(dim//2,device=t.device,dtype=torch.float32)/(dim//2))
    phase = t.float()[...,None] * 1000 * frequencies
    return torch.cat([phase.cos(), phase.sin()], -1)

def rope(x, positions):
    freq = torch.exp(-math.log(10000) * torch.arange(0,x.shape[-1],2,device=x.device,dtype=torch.float32)/x.shape[-1])
    phase = positions.float()[:,None,:,None]*freq
    c,s = phase.cos().to(x.dtype),phase.sin().to(x.dtype)
    a,b = x[...,::2],x[...,1::2]
    return torch.stack([a*c-b*s,a*s+b*c],-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width,3*width,bias=False)
        self.out = nn.Linear(width,width,bias=False)
    def forward(self,x,positions,allowed,past=None):
        b,n,w = x.shape
        q,k,v = self.qkv(x).view(b,n,3,self.heads,w//self.heads).permute(2,0,3,1,4).unbind(0)
        q,k = rope(q,positions),rope(k,positions)
        if past is not None:
            k,v = torch.cat([past[0],k],-2),torch.cat([past[1],v],-2)
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.0)
        return self.out(y.transpose(1,2).reshape(b,n,w)), (k,v)

class Block(nn.Module):
    def __init__(self,width,heads,ffn,fp32_residual=False):
        super().__init__()
        self.fp32_residual = fp32_residual
        self.norm1,self.norm2 = RMSNorm(width),RMSNorm(width)
        self.attn = Attention(width,heads)
        self.up = nn.Linear(width,2*ffn,bias=False)
        self.down = nn.Linear(ffn,width,bias=False)
        self.mod = nn.Linear(width,6*width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
    def forward(self,x,condition,positions,allowed,past=None):
        s1,a1,g1,s2,a2,g2 = self.mod(F.silu(condition)).chunk(6,-1)
        h,kv = self.attn(self.norm1(x)*(1+a1)+s1,positions,allowed,past)
        # GMM training can grow the gated residual beyond FP16's finite range.
        # Keep its multiplication and accumulation wide; GEMMs still use AMP.
        x = x.float()+g1.float()*h.float() if self.fp32_residual else x+g1*h
        u,v = self.up(self.norm2(x)*(1+a2)+s2).chunk(2,-1)
        h = self.down(F.silu(u)*v)
        return (x+g2.float()*h.float() if self.fp32_residual else x+g2*h),kv

class Denoiser(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = dict(cfg)
        w,d = cfg["width"],cfg.get("latent_dim",48)
        if w % cfg["heads"] or (w//cfg["heads"])%2:
            raise ValueError("RoPE requires an even head width")
        self.input = nn.Linear(d,w,bias=False)
        self.roles = nn.Embedding(4,w)
        capacity=cfg.get("length_capacity",cfg["block"])
        if cfg["block"]<1 or capacity<cfg["block"]:
            raise ValueError("Length embedding capacity must cover the diffusion block")
        self.lengths = nn.Embedding(capacity+1,w)
        self.time = nn.Sequential(nn.Linear(256,w),nn.SiLU(),nn.Linear(w,w))
        self.blocks = nn.ModuleList([Block(w,cfg["heads"],cfg["ffn"],fp32_residual=bool(cfg.get("gmm_components",0))) for _ in range(cfg["layers"])])
        self.norm = RMSNorm(w)
        self.final_mod = nn.Linear(w,2*w)
        components = int(cfg.get("gmm_components", 0))
        if components < 0:
            raise ValueError("gmm_components must be nonnegative")
        if components and (not math.isfinite(cfg["gmm_sigma"]) or cfg["gmm_sigma"] <= 0):
            raise ValueError("GMM requires a finite positive fixed sigma")
        self.output = nn.Linear(w,components*(d+1) if components else d)
        nn.init.zeros_(self.final_mod.weight)
        nn.init.zeros_(self.final_mod.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,x,t,roles,positions,lengths,allowed,past=None,return_cache=False):
        h = self.input(x)
        role = self.roles(roles).to(h.dtype)
        h = h+role+self.lengths(lengths).to(h.dtype)
        condition = self.time(timestep_embedding(t)).to(h.dtype)+role
        caches=[]
        for i,block in enumerate(self.blocks):
            pkv = None if past is None else past[i]
            if self.training and self.cfg.get("grad_checkpoint",False) and not return_cache:
                def run(value,cond,layer=block):
                    return layer(value,cond,positions,allowed)[0]
                h = checkpoint(run,h,condition,use_reentrant=False)
            else:
                h,kv = block(h,condition,positions,allowed,pkv)
                if return_cache:
                    caches.append(kv)
        shift,scale=self.final_mod(F.silu(condition)).chunk(2,-1)
        result=self.output(self.norm(h)*(1+scale)+shift)
        return (result,caches) if return_cache else result
