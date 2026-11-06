from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .model import CLEAN,NOISY,BOS,PAD
from .utils import sha256

class LatentCache:
    def __init__(self,path,split="train",verify_hashes=False,block=None):
        self.root=Path(path)
        self.manifest=json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        self.fingerprint=sha256(self.root/"manifest.json")
        m=self.manifest
        for name,meta in m["files"].items():
            p=self.root/name
            if p.stat().st_size!=meta["bytes"]:raise ValueError(f"Cache file size mismatch: {name}")
            if verify_hashes and sha256(p)!=meta["sha256"]:raise ValueError(f"Cache digest mismatch: {name}")
        self.mean=np.asarray(m["normalizer"]["mean"],np.float32)
        self.scale=float(m["normalizer"]["scale"])
        v=np.fromfile(self.root/"vectors.f32",dtype="<f4").reshape(-1,48)
        self.vectors=(v-self.mean)/self.scale
        def mmap(name,dtype):
            p=self.root/name
            return np.memmap(p,mode="r",dtype=dtype) if p.stat().st_size else np.empty(0,dtype=dtype)
        self.ids=mmap(f"{split}.ids.u32","<u4")
        self.offsets=mmap(f"{split}.offsets.u64","<u8")
        self.segments=mmap(f"{split}.segments.i64","<i8").reshape(-1,4)
        self.windows=mmap(f"{split}.windows.u64","<u8")
        self.valid=mmap(f"{split}.valid.u32","<u4")
        self.packing_block=int(m["layout"]["block"])
        self.block=self.packing_block if block is None else block
        if not isinstance(self.block,int) or self.block<1 or self.packing_block%self.block:
            raise ValueError("Effective diffusion block must be a positive divisor of the cache packing block")
        self.units=int(m["splits"][split]["units"])
        self.split=split
        if int(self.valid.astype(np.uint64).sum())!=self.units:raise ValueError("Window targets do not cover the corpus exactly")
    def __len__(self):return len(self.windows)-1
    def story(self,doc):
        return self.vectors[self.ids[int(self.offsets[doc]):int(self.offsets[doc+1])]]
    def target_units(self,indices):return int(self.valid[np.asarray(indices)].astype(np.uint64).sum())
    def layout(self,index):
        clean=[];noisy=[];block_id=0
        for doc_number,part in enumerate(self.segments[int(self.windows[index]):int(self.windows[index+1])]):
            doc,st,en,pr=map(int,part);base=int(self.offsets[doc])
            clean.append((np.zeros(48,np.float32),BOS,pr,doc_number,-1,-1,0))
            for i in range(pr,en):
                clean.append((self.vectors[int(self.ids[base+i])],CLEAN,i+1,doc_number,-1,-1,0))
            for start in range(st,en,self.block):
                length=min(self.block,en-start)
                for i in range(start,start+length):
                    noisy.append((self.vectors[int(self.ids[base+i])],NOISY,i+1,doc_number,block_id,start+1,length))
                block_id+=1
        return clean+noisy,block_id
    def batch(self,indices,device="cpu"):
        layouts=[self.layout(i) for i in indices]
        n=max(len(v[0]) for v in layouts);b=len(layouts)
        x=np.zeros((b,n,48),np.float32);roles=np.full((b,n),PAD,np.int64)
        pos=np.zeros((b,n),np.int64);docs=np.full((b,n),-1,np.int64)
        blocks=np.full((b,n),-1,np.int64);starts=np.full((b,n),-1,np.int64);lengths=np.zeros((b,n),np.int64)
        offset=0
        for i,(items,nb) in enumerate(layouts):
            for j,(z,role,p,doc,block,start,length) in enumerate(items):
                x[i,j]=z;roles[i,j]=role;pos[i,j]=p;docs[i,j]=doc
                blocks[i,j]=block+offset if block>=0 else -1;starts[i,j]=start;lengths[i,j]=length
            offset+=nb
        batch={k:torch.from_numpy(v).to(device) for k,v in
               dict(x=x,roles=roles,positions=pos,docs=docs,noise_block=blocks,starts=starts,lengths=lengths).items()}
        batch["allowed"]=attention_mask(batch)
        return batch

def attention_mask(batch):
    r,p,d,b,s=[batch[k] for k in ("roles","positions","docs","noise_block","starts")]
    clean=(r==CLEAN)|(r==BOS);noisy=r==NOISY
    same=(d[:,:,None]==d[:,None,:])&(d[:,:,None]>=0)
    allowed_clean=clean[:,:,None]&clean[:,None,:]&(p[:,None,:]<=p[:,:,None])
    allowed_history=noisy[:,:,None]&clean[:,None,:]&(p[:,None,:]<s[:,:,None])
    allowed_noisy=noisy[:,:,None]&noisy[:,None,:]&(b[:,:,None]==b[:,None,:])
    allow=same&(allowed_clean|allowed_history|allowed_noisy)
    eye=torch.eye(r.shape[1],device=r.device,dtype=torch.bool)[None]
    allow=allow|((r==PAD)[:,:,None]&eye)
    return allow[:,None]

class WindowSampler:
    def __init__(self,size,seed=1234):
        if size<1:raise ValueError("No training windows")
        self.size=size;self.seed=seed;self.epoch=0;self.cursor=0;self._order=None
    def next(self,n):
        result=[]
        while len(result)<n:
            if self._order is None:self._order=np.random.default_rng(self.seed+self.epoch).permutation(self.size)
            count=min(n-len(result),self.size-self.cursor)
            result.extend(self._order[self.cursor:self.cursor+count].tolist());self.cursor+=count
            if self.cursor==self.size:self.epoch+=1;self.cursor=0;self._order=None
        return result
    def state_dict(self):return {"epoch":self.epoch,"cursor":self.cursor,"size":self.size,"seed":self.seed}
    def load_state_dict(self,state):
        if state["size"]!=self.size or state["seed"]!=self.seed:raise ValueError("Sampler configuration mismatch")
        self.epoch=state["epoch"];self.cursor=state["cursor"];self._order=None
