"""One-pass multilingual units, bounded FP16 caching, and mixed-block layouts."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
import queue
import threading

import numpy as np
import pyarrow.parquet as pq
import torch

from .bpe_packing import BPEPacker
from .data import attention_mask
from .model import BOS, CLEAN, NOISY, PAD
from .segmented_cache import FP16ShardCache, fingerprint, read_segmented_shard
from .utils import sha256, write_json


class RawCorpus:
    def __init__(self, manifest, seed=1234):
        self.path=Path(manifest)
        self.spec=json.loads(self.path.read_text(encoding='utf8'))
        if self.spec['format']!='full-raw-ae-v1' or self.spec['status']!='complete':
            raise ValueError('Expected the complete admitted raw corpus')
        self.seed=seed; self.groups=[]; self.readers={}
        self.fingerprint=sha256(self.path)
        for pi,part in enumerate(self.spec['parts']):
            offset=0
            for gi,size in enumerate(part['groups']):
                if size:self.groups.append((pi,gi,offset,size,part['language']))
                offset+=size
        self.order=np.random.default_rng(seed).permutation(len(self.groups)).tolist()

    def read(self, index):
        pi,gi,offset,size,lang=self.groups[index]
        part=self.spec['parts'][pi]
        if pi not in self.readers:self.readers[pi]=pq.ParquetFile(self.path.parent/part['file'])
        texts=self.readers[pi].read_row_group(gi,columns=['story'],use_threads=True).column(0).to_pylist()
        if len(texts)!=size:raise ValueError('Raw row-group count changed')
        return texts,[(pi,offset+i) for i in range(size)],lang

    def close(self):
        for reader in self.readers.values():reader.close()


class GridUnits:
    def __init__(self, directory, encoder, hot, max_bytes, ae_sha256, source_fingerprint,
                 require_complete=True):
        self.directory=Path(directory); self.encoder=encoder; self.loaded=OrderedDict()
        progress=json.loads((self.directory/'progress.json').read_text(encoding='utf8'))['state']
        if require_complete and not progress['complete']:raise ValueError('Grid8 segmentation is still running')
        self.paths=[self.directory/'segments'/f'shard-{i:06d}' for i in range(progress['shards'])]
        self.starts=[]
        self.provenance=json.loads((self.directory/'segmentation.json').read_text(encoding='utf8'))
        # Bind the actual production protocol, including its frozen AE fingerprint.
        if ae_sha256 not in json.dumps(self.provenance):raise ValueError('Grid8 AE checkpoint differs')
        for path in self.paths:
            meta=json.loads((path/'manifest.json').read_text(encoding='utf8'))
            self.starts.append(tuple(meta['metadata']['before_cursor']))
        self.cache=FP16ShardCache(hot,dict(ae_sha256=ae_sha256,source=source_fingerprint,
                 segmentation=self.provenance,encoding='fp32-length-bucket-to-fp16-v1'),max_bytes)

    def units(self, text, coordinate):
        index=bisect_right(self.starts,coordinate)-1
        if index<0:raise ValueError('Missing grid8 source coordinate')
        if index not in self.loaded:
            shard=read_segmented_shard(self.paths[index])
            lookup={(int(f),int(r)):i for i,(f,r) in enumerate(zip(shard['source_files'],shard['source_rows']))}
            self.loaded[index]=(shard,lookup)
            if len(self.loaded)>4:self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        shard,lookup=self.loaded[index]
        if coordinate not in lookup:raise ValueError('Raw document missing from grid8 segmentation')
        i=lookup[coordinate]
        lo,hi=map(int,shard['byte_offsets'][i:i+2]); a,b=map(int,shard['unit_offsets'][i:i+2])
        raw=text.encode()
        if shard['text'][lo:hi]!=raw:raise ValueError('Grid8/raw document mismatch')
        lengths=shard['lengths'][a:b]
        ends=np.cumsum(lengths,dtype=np.int64); starts=np.r_[0,ends[:-1]]
        return [raw[int(s):int(e)] for s,e in zip(starts,ends)]

    def group(self,index,texts,coordinates):
        units=[self.units(t,c) for t,c in zip(texts,coordinates)]
        flat=[u for row in units for u in row]
        def build(path):
            with Path(path).open('wb') as handle:
                for start in range(0,len(flat),8192):
                    self.encoder.encode(flat[start:start+8192])[0].tofile(handle)
        path=self.cache.ensure(f'group-{index:06d}',len(flat),build)
        vectors=np.fromfile(path,dtype='<f2').reshape(-1,48)
        result=[]; offset=0
        for row in units:
            result.append((row,vectors[offset:offset+len(row)])); offset+=len(row)
        return result


class GroupProvider:
    def __init__(self,cfg,encoder,root):
        self.cfg=cfg; self.root=Path(root); dc=cfg['data']
        self.raw=RawCorpus(self.root/dc['manifest'],cfg['train']['seed'])
        self.encoder=encoder
        if dc['mode']=='bpe':
            self.packer=BPEPacker(self.root/dc['bpe_assets'],encoder,dc['result_cache_entries'])
            self.protocol=dict(raw=self.raw.fingerprint,mode='bpe',packing=self.packer.manifest['fingerprint'])
            self.grid=None
        elif dc['mode']=='grid8':
            self.grid=GridUnits(self.root/dc['segments'],encoder,self.root/dc['hot_cache'],
                int(dc['hot_cache_gib']*1024**3),dc['ae_sha256'],self.raw.fingerprint,
                require_complete=dc.get('require_complete',True))
            self.protocol=dict(raw=self.raw.fingerprint,mode='grid8',segmentation=self.grid.provenance)
            self.packer=None
        else:raise ValueError('Unknown prior data mode')
        self.protocol.update(seed=cfg['train']['seed'],window=dc['window'],carry=dc['carry'],
                             lanes=dc['lanes'],max_documents_per_window=64,
                             order='shuffled-groups-and-documents-round-robin-v1')
        self.fingerprint=fingerprint(self.protocol)

    def load(self,index):
        texts,coords,lang=self.raw.read(index)
        if self.packer:
            rows=[]
            for start in range(0,len(texts),128):rows.extend(self.packer.pack(texts[start:start+128]))
        else:rows=self.grid.group(index,texts,coords)
        order=np.random.default_rng(np.random.SeedSequence([self.raw.seed,index,7919])).permutation(len(rows))
        return [(rows[i][0],rows[i][1],lang,coords[i]) for i in order]

    def normalizer(self,path):
        path=Path(path)
        if path.exists():
            value=json.loads(path.read_text(encoding='utf8'))
            if value['data_fingerprint']!=self.fingerprint:raise ValueError('Normalizer data differs')
            return value
        by_language={}
        for i,g in enumerate(self.raw.groups):by_language.setdefault(g[-1],[]).append(i)
        weights={}
        for p in self.raw.spec['parts']:
            weights[p['language']]=weights.get(p['language'],0)+p['counts']['raw_bytes']
        total=0.; sums=np.zeros(48,np.float64); squares=0.; panel=[]
        rng=np.random.default_rng(9137)
        for lang,indices in sorted(by_language.items()):
            selected=int(rng.choice(indices)); rows=self.load(selected)
            count=sum(len(row[0]) for row in rows); raw_bytes=sum(sum(map(len,row[0])) for row in rows)
            weight=weights[lang]/raw_bytes
            for _,z,_,_ in rows:
                z=z.astype(np.float64); sums+=weight*z.sum(0); squares+=weight*np.square(z).sum()
            total+=weight*count
            panel.append(dict(language=lang,group=selected,units=count,raw_bytes=raw_bytes,weight=weight))
        mean=sums/total; scale=float(np.sqrt(squares/(48*total)-np.square(mean).mean()))
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid scalar normalizer')
        value=dict(data_fingerprint=self.fingerprint,mean=mean.tolist(),scale=scale,panel=panel,
                   protocol='one seeded row-group per language; weighted by full raw bytes; FP16 cache coordinates')
        write_json(path,value)
        return value

    def close(self):self.raw.close()


class UnitStream:
    def __init__(self,provider,block_sizes=(1,4,8)):
        self.provider=provider; self.block_sizes=tuple(block_sizes)
        dc=provider.cfg['data']; self.window=dc['window']; self.carry=dc['carry']
        n=min(dc['lanes'],len(provider.raw.order))
        self.lanes=[[g,0,0] for g in provider.raw.order[:n]]
        self.next_group=n; self.turn=0; self.loaded={}
        self.rng=np.random.default_rng(provider.raw.seed+104729)

    def state_dict(self):
        return dict(fingerprint=self.provider.fingerprint,lanes=deepcopy(self.lanes),next_group=self.next_group,
                    turn=self.turn,rng=deepcopy(self.rng.bit_generator.state),blocks=list(self.block_sizes))

    def load_state_dict(self,state):
        if state['fingerprint']!=self.provider.fingerprint or state['blocks']!=list(self.block_sizes):
            raise ValueError('Incompatible unit stream checkpoint')
        self.lanes=deepcopy(state['lanes']); self.next_group=state['next_group']; self.turn=state['turn']
        self.rng.bit_generator.state=state['rng']; self.loaded.clear()

    def next_window(self,block):
        parts=[]; remaining=self.window
        while remaining and len(parts)<64 and any(lane is not None for lane in self.lanes):
            while self.lanes[self.turn] is None:self.turn=(self.turn+1)%len(self.lanes)
            group,row,start=self.lanes[self.turn]
            if group not in self.loaded:self.loaded[group]=self.provider.load(group)
            units,z,lang,coordinate=self.loaded[group][row]
            end=min(len(units),start+remaining); prior=max(0,start-self.carry)
            phase=int(self.rng.integers(block))
            parts.append(dict(z=z[prior:end].copy(),prior=prior,start=start,end=end,phase=phase,
                              lengths=np.asarray([len(u) for u in units[start:end]],np.uint8),
                              language=lang,coordinate=coordinate))
            remaining-=end-start
            if end==len(units):
                row+=1
                if row==len(self.loaded[group]):
                    del self.loaded[group]
                    if self.next_group<len(self.provider.raw.order):
                        self.lanes[self.turn]=[self.provider.raw.order[self.next_group],0,0]; self.next_group+=1
                    else:self.lanes[self.turn]=None
                else:self.lanes[self.turn]=[group,row,0]
                self.turn=(self.turn+1)%len(self.lanes)
            else:self.lanes[self.turn]=[group,row,end]
        return parts

    def next_batch(self,windows):
        block=int(self.rng.choice(self.block_sizes))
        rows=[]
        for _ in range(windows):
            row=self.next_window(block)
            if not row:break
            rows.append(row)
        if not rows:return None
        count=sum(p['end']-p['start'] for row in rows for p in row)
        raw_bytes=sum(int(p['lengths'].sum()) for row in rows for p in row)
        languages={}
        for row in rows:
            for p in row:languages[p['language']]=languages.get(p['language'],0)+int(p['lengths'].sum())
        return dict(windows=rows,block=block,units=count,raw_bytes=raw_bytes,languages=languages,
                    after=self.state_dict())


def collate(windows,block,normalizer,device='cpu'):
    sizes=[sum(1+len(p['z'])+p['end']-p['start'] for p in row) for row in windows]
    b,n=len(windows),max(sizes)
    arrays=dict(x=np.zeros((b,n,48),np.float32),roles=np.full((b,n),PAD,np.int64),
                positions=np.zeros((b,n),np.int64),docs=np.full((b,n),-1,np.int64),
                noise_block=np.full((b,n),-1,np.int64),starts=np.full((b,n),-1,np.int64),
                lengths=np.zeros((b,n),np.int64))
    mean=np.asarray(normalizer['mean'],np.float32); scale=normalizer['scale']; bid=0
    for wi,row in enumerate(windows):
        clean_end=sum(1+len(p['z']) for p in row); c=0; q=clean_end
        for di,p in enumerate(row):
            prior,start,end=p['prior'],p['start'],p['end']; z=(p['z'].astype(np.float32)-mean)/scale
            size=len(z); target=end-start
            arrays['roles'][wi,c]=BOS; arrays['positions'][wi,c]=prior; arrays['docs'][wi,c]=di; c+=1
            sl=slice(c,c+size)
            arrays['x'][wi,sl]=z; arrays['roles'][wi,sl]=CLEAN
            arrays['positions'][wi,sl]=np.arange(prior+1,end+1); arrays['docs'][wi,sl]=di; c+=size
            sl=slice(q,q+target)
            arrays['x'][wi,sl]=z[start-prior:]; arrays['roles'][wi,sl]=NOISY
            arrays['positions'][wi,sl]=np.arange(start+1,end+1); arrays['docs'][wi,sl]=di
            cursor=start; phase=p['phase']
            while cursor<end:
                length=min(phase if cursor==start and phase else block,end-cursor)
                sl=slice(q+cursor-start,q+cursor-start+length)
                arrays['noise_block'][wi,sl]=bid; arrays['starts'][wi,sl]=cursor+1
                arrays['lengths'][wi,sl]=length; bid+=1; cursor+=length
            q+=target
    batch={k:torch.from_numpy(v).to(device) for k,v in arrays.items()}
    batch['allowed']=attention_mask(batch)
    return batch


class PrefetchedUnits:
    """A separate CUDA stream prepares frozen latents; commit only learned data."""
    def __init__(self,source,windows,depth=3):
        self.source=source; self.windows=windows; self.queue=queue.Queue(depth)
        self.stop=threading.Event(); self.thread=None; self.committed=source.state_dict()

    def _put(self,value):
        while not self.stop.is_set():
            try:self.queue.put(value,timeout=.2);return
            except queue.Full:pass

    def _produce(self):
        try:
            device=self.source.provider.encoder.device
            context=torch.cuda.stream(torch.cuda.Stream(device=device)) if device.type=='cuda' else torch.inference_mode()
            with context,torch.inference_mode():
                while not self.stop.is_set():
                    batch=self.source.next_batch(self.windows)
                    self._put(batch)
                    if batch is None:break
        except BaseException as exc:self._put(exc)

    def next(self):
        if self.thread is None:
            self.thread=threading.Thread(target=self._produce,daemon=True); self.thread.start()
        value=self.queue.get()
        if isinstance(value,BaseException):raise value
        return value

    def commit(self,batch):self.committed=deepcopy(batch['after'])
    def state_dict(self):return deepcopy(self.committed)
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=30)
        if not self.thread or not self.thread.is_alive():self.source.provider.close()
