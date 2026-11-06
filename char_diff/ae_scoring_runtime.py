"""Frozen FP32 AE inference with length buckets and shared decoder work."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F

from .ae_nll import AENLLScorer,tensorize_compact
from .ae_quality import targets_from_lengths
from .ae_paths import CandidateScore


class OptimizedAENLLScorer(AENLLScorer):
    """Same mathematical byte+EOS score, with explicit numerical audit required.

    The first decoder projection and FiLM linear precede position conditioning;
    compute them once per unit. Frozen position modulations are computed once.
    Optional encoder buckets omit padding queries and unused decoder positions.
    """
    def __init__(self,ae,*,device=None,batch_size=2048,bucketed=True,factored=True):
        super().__init__(ae,device=device,batch_size=batch_size,precision='fp32')
        self.bucketed=bucketed;self.factored=factored
        with torch.inference_mode():
            decoder=self.ae.decoder
            delta=decoder.delta_emb(torch.arange(decoder.k_max,device=self.device))
            self.modulations=[tuple(t.detach() for t in film.mod(delta).chunk(2,-1)) for film in decoder.films]

    def decode(self,z,k):
        d=self.ae.decoder
        if not self.factored:
            return d(z.repeat_interleave(k,0),torch.arange(k,device=self.device).repeat(len(z))).view(len(z),k,257)
        x=F.silu(d.in_proj(z))
        for index,film in enumerate(d.films):
            h=film.fc(F.rms_norm(x,(x.shape[-1],),film.norm.weight,film.norm.eps))
            if index==0:h=h[:,None,:]
            g,b=self.modulations[index]
            x=F.silu(torch.addcmul(b[None,:k],h,g[None,:k]).add(h))
        if not d.films:x=x[:,None,:].expand(-1,k,-1)
        for fc in d.rest:x=F.silu(fc(x))
        return d.out(x)

    @torch.inference_mode()
    def __call__(self,units):
        output=[None]*len(units);groups=defaultdict(list)
        for i,u in enumerate(units):
            upper=min(self.l_max,((len(u)+7)//8)*8) if self.bucketed else self.l_max
            groups[upper].append(i)
        was_training=self.ae.training;self.ae.eval()
        try:
            with torch.autocast(device_type=self.device.type,enabled=False):
                for upper,indices in groups.items():
                    for start in range(0,len(indices),self.batch_size):
                        selected=indices[start:start+self.batch_size];part=[units[i] for i in selected]
                        ids,lens,mask=tensorize_compact(part,upper,self.device)
                        targets=targets_from_lengths(ids,lens);valid=targets>=0
                        logits=self.decode(self.ae.encoder(ids,mask),upper+1).float()
                        logp=logits.log_softmax(-1)
                        nll=-logp.gather(-1,targets.clamp_min(0)[...,None]).squeeze(-1)
                        nll=nll.masked_fill(~valid,0).sum(-1)
                        feasible=((logits.argmax(-1)==targets)|~valid).all(-1)
                        finite=torch.isfinite(logits).all(dim=(1,2))
                        values=torch.stack((feasible.float(),nll,finite.float()),-1).cpu().tolist()
                        for index,(ok,q,safe) in zip(selected,values):
                            if not safe:raise FloatingPointError('Non-finite optimized AE logits')
                            output[index]=CandidateScore(bool(ok),q)
        finally:self.ae.train(was_training)
        return output
