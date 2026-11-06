"""The existing FP32 encoder/decoder with compact NumPy input and output."""
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import torch
from .ae_scoring_runtime import OptimizedAENLLScorer
from .ae_quality import targets_from_lengths
from .ae_paths import CandidateScore


class PackedAENLLScorer(OptimizedAENLLScorer):
    @torch.inference_mode()
    def score_packed(self,packed):
        if packed.dtype!=np.uint8 or packed.ndim!=2 or packed.shape[1]!=33:raise ValueError('Expected uint8 (N,33) candidates')
        if self.device.type!='cuda':raise ValueError('Packed scoring expects CUDA')
        lengths=packed[:,0]
        if ((lengths<1)|(lengths>self.l_max)).any():raise ValueError('Invalid packed byte lengths')
        output=np.empty((len(packed),2),np.float64);pending=None
        was_training=self.ae.training;self.ae.eval()
        def finish(item):
            indices,host,event=item;event.synchronize();values=host.numpy()
            if not values[:,2].all():raise FloatingPointError('Non-finite packed AE logits')
            output[indices]=values[:,:2]
        try:
            with torch.autocast(device_type='cuda',enabled=False):
                buckets=((lengths.astype(np.int64)+7)//8)*8
                for upper in np.unique(buckets):
                    upper=int(upper)
                    indices=np.flatnonzero(buckets==upper)
                    for at in range(0,len(indices),self.batch_size):
                        selected=indices[at:at+self.batch_size]
                        host_input=torch.empty((len(selected),33),dtype=torch.uint8,pin_memory=True)
                        host_input.copy_(torch.from_numpy(packed[selected]))
                        inputs=host_input.to(self.device,non_blocking=True)
                        ids=inputs[:,1:upper+1].long();lens=inputs[:,0].long()
                        mask=torch.arange(upper,device=self.device)[None]>=lens[:,None]
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
                        previous=pending;pending=(selected,host,event)
                        if previous is not None:finish(previous)
                if pending is not None:finish(pending)
        finally:self.ae.train(was_training)
        return output


class ConcurrentPackedScorer:
    def __init__(self,ae,*,device='cuda',batch_size=8192,streams=2):
        if streams<1:raise ValueError('Expected positive stream count')
        ae.eval()
        self.scorers=[PackedAENLLScorer(ae,device=device,batch_size=batch_size) for _ in range(streams)]
        self.ready=torch.cuda.Event();self.ready.record();self.local=threading.local()
        self.lock=threading.Lock();self.next_worker=0
        self.pool=ThreadPoolExecutor(max_workers=streams,thread_name_prefix='packed-cuda')
        self.chunk=max(batch_size*4,65536)
    def _score(self,packed):
        if not hasattr(self.local,'index'):
            with self.lock:self.local.index=self.next_worker;self.next_worker+=1
            self.local.stream=torch.cuda.Stream();self.local.stream.wait_event(self.ready)
        with torch.cuda.stream(self.local.stream):return self.scorers[self.local.index].score_packed(packed)
    def score_packed(self,packed):
        if not len(packed):return np.empty((0,2),np.float64)
        return np.concatenate(list(self.pool.map(self._score,
                     (packed[i:i+self.chunk] for i in range(0,len(packed),self.chunk)))))
    def __call__(self,units):
        from .coarse_process import pack_candidates
        return [CandidateScore(bool(ok),q) for ok,q in self.score_packed(pack_candidates(units))]
    def close(self):self.pool.shutdown(wait=True)
