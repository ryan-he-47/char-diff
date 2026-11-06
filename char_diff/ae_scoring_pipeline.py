"""Overlap the next frozen FP32 batch with CPU unpacking of the previous one."""
from collections import defaultdict
import numpy as np
import torch
from .ae_scoring_runtime import OptimizedAENLLScorer
from .ae_quality import targets_from_lengths
from .ae_paths import CandidateScore
from concurrent.futures import ThreadPoolExecutor
import threading


class PipelinedAENLLScorer(OptimizedAENLLScorer):
    def _inputs(self,units,upper):
        for unit in units:
            if not isinstance(unit,bytes) or not 1<=len(unit)<=upper:raise ValueError('Invalid candidate')
            unit.decode('utf8',errors='strict')
        packed=b''.join(unit.ljust(upper,b'\0') for unit in units)
        host=torch.empty((len(units),upper),dtype=torch.uint8,pin_memory=True)
        host.copy_(torch.from_numpy(np.frombuffer(packed,np.uint8).reshape(len(units),upper).copy()))
        lengths=torch.tensor([len(u) for u in units],dtype=torch.long,pin_memory=True)
        ids=host.to(self.device,non_blocking=True).long()
        lens=lengths.to(self.device,non_blocking=True)
        return ids,lens,torch.arange(upper,device=self.device)[None]>=lens[:,None]

    @torch.inference_mode()
    def __call__(self,units):
        if self.device.type!='cuda':return super().__call__(units)
        groups=defaultdict(list);output=[None]*len(units)
        for i,u in enumerate(units):groups[min(self.l_max,((len(u)+7)//8)*8)].append(i)
        was_training=self.ae.training;self.ae.eval();pending=None
        def finish(item):
            selected,host,event=item;event.synchronize()
            for index,(ok,q,safe) in zip(selected,host.tolist()):
                if not safe:raise FloatingPointError('Non-finite pipelined AE logits')
                output[index]=CandidateScore(bool(ok),q)
        try:
            with torch.autocast(device_type='cuda',enabled=False):
                for upper,indices in groups.items():
                    for start in range(0,len(indices),self.batch_size):
                        selected=indices[start:start+self.batch_size]
                        ids,lens,mask=self._inputs([units[i] for i in selected],upper)
                        targets=targets_from_lengths(ids,lens);valid=targets>=0
                        logits=self.decode(self.ae.encoder(ids,mask),upper+1).float()
                        logp=logits.log_softmax(-1)
                        nll=-logp.gather(-1,targets.clamp_min(0)[...,None]).squeeze(-1)
                        nll=nll.masked_fill(~valid,0).sum(-1)
                        feasible=((logits.argmax(-1)==targets)|~valid).all(-1)
                        finite=torch.isfinite(logits).all(dim=(1,2))
                        values=torch.stack((feasible.float(),nll,finite.float()),-1)
                        host=torch.empty(values.shape,dtype=values.dtype,pin_memory=True)
                        host.copy_(values,non_blocking=True);event=torch.cuda.Event();event.record()
                        # Launch the next batch before waiting for/unpacking its predecessor.
                        previous=pending;pending=(selected,host,event)
                        if previous is not None:finish(previous)
                if pending is not None:finish(pending)
        finally:self.ae.train(was_training)
        return output


class ConcurrentAENLLScorer:
    """Independent CUDA streams share read-only AE weights and bounded batches."""
    def __init__(self,ae,*,device='cuda',batch_size=16384,streams=2):
        if streams<1:raise ValueError('Expected positive stream count')
        ae.eval()
        self.scorers=[PipelinedAENLLScorer(ae,device=device,batch_size=batch_size) for _ in range(streams)]
        self.ready=torch.cuda.Event();self.ready.record()
        self.local=threading.local();self.lock=threading.Lock();self.next_worker=0
        self.pool=ThreadPoolExecutor(max_workers=streams,thread_name_prefix='ae-cuda')
        self.chunk=max(batch_size*4,65536)
    def _score(self,units):
        if not hasattr(self.local,'index'):
            with self.lock:self.local.index=self.next_worker;self.next_worker+=1
            self.local.stream=torch.cuda.Stream();self.local.stream.wait_event(self.ready)
        with torch.cuda.stream(self.local.stream):return self.scorers[self.local.index](units)
    def __call__(self,units):
        return [score for result in self.pool.map(self._score,
                 (units[i:i+self.chunk] for i in range(0,len(units),self.chunk))) for score in result]
    def close(self):self.pool.shutdown(wait=True)
