"""UTF-8 BPE triplets, frozen FP32 verification, and bounded result caching."""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .ae_nll import tensorize_compact
from .ae_scoring_runtime import OptimizedAENLLScorer
from .segmented_cache import fingerprint
from .utils import sha256, write_json
from .vendor.bltz.config import Cfg
from .vendor.bltz.models.autoencoder import ByteStringAE


def valid_utf8(raw):
    try:
        raw.decode('utf8')
        return bool(raw)
    except UnicodeDecodeError:
        return False


def byte_inverse():
    visible = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    extra = [b for b in range(256) if b not in visible]
    return {chr(c): b for b,c in zip(visible+extra, visible+list(range(256,256+len(extra))))}


class FrozenUnits:
    def __init__(self, checkpoint, device='cuda', batch_size=1024):
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ae = ByteStringAE(Cfg(state['cfg']))
        ae.load_state_dict(state['model'])
        self.scorer = OptimizedAENLLScorer(ae.eval().requires_grad_(False), device=device,
                                           batch_size=batch_size)
        self.device = torch.device(device)
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, units, score=False):
        vectors = np.empty((len(units),48), np.float16)
        quality = np.empty((len(units),2), np.float64) if score else None
        lengths = np.asarray([len(u) for u in units])
        if np.any((lengths < 1) | (lengths > 32)):
            raise ValueError('Frozen AE units must have 1..32 bytes')
        with torch.autocast(device_type=self.device.type, enabled=False):
            for upper in (8,16,24,32):
                indices = np.flatnonzero((lengths > upper-8) & (lengths <= upper))
                for start in range(0,len(indices),self.batch_size):
                    ix = indices[start:start+self.batch_size]
                    part=[units[i] for i in ix]
                    # A fixed GEMM shape makes cache misses/restarts use the same
                    # frozen numerical path, including the final short batch.
                    part+=part[:1]*(self.batch_size-len(part))
                    ids,lens,mask = tensorize_compact(part, upper, self.device)
                    z = self.scorer.ae.encoder(ids,mask)
                    if not torch.isfinite(z).all():
                        raise FloatingPointError('Non-finite frozen AE encoding')
                    vectors[ix] = z[:len(ix)].cpu().numpy().astype(np.float16)
                    if score:
                        logits = self.scorer.decode(z,upper+1).float()
                        target = torch.nn.functional.pad(ids,(0,1))
                        target[torch.arange(len(ids),device=self.device),lens] = 256
                        live = torch.arange(upper+1,device=self.device)[None] <= lens[:,None]
                        nll = -logits.log_softmax(-1).gather(2,target[:,:,None]).squeeze(2)
                        nll = nll.masked_fill(~live,0).sum(1)
                        exact = ((logits.argmax(-1)==target)|~live).all(1)
                        if not torch.isfinite(nll).all():
                            raise FloatingPointError('Non-finite frozen AE verification')
                        quality[ix] = torch.stack((exact,nll),1)[:len(ix)].cpu().numpy()
        if not np.isfinite(vectors).all():
            raise FloatingPointError('FP16 latent cache overflow')
        return vectors,quality


def prepare_assets(tokenizer_path, checkpoint, scores_path, out, encoder, threshold=.03):
    """Reuse the measured vocabulary scores; encode each static string once."""
    out = Path(out)
    protocol = dict(version='bpe-triplet-v1', ae_sha256=sha256(checkpoint),
                    tokenizer_sha256=sha256(tokenizer_path), threshold=threshold,
                    precision='fp32', normalization='none', utf8='complete codepoints',
                    fixed_encoder_batch=encoder.batch_size)
    if (out/'manifest.json').exists():
        meta = json.loads((out/'manifest.json').read_text(encoding='utf8'))
        if meta['protocol'] != protocol:
            raise ValueError('BPE asset protocol mismatch')
        return meta
    out.mkdir(parents=True,exist_ok=True)
    spec = json.loads(Path(tokenizer_path).read_text(encoding='utf8'))
    inverse = byte_inverse()
    vocab = {word:bytes(inverse[c] for c in word) for word in spec['model']['vocab']}
    units = [u for u in vocab.values() if len(u)<=32 and valid_utf8(u)]
    saved = np.load(scores_path)
    if (str(saved['checkpoint_sha256'])!=protocol['ae_sha256'] or
            str(saved['tokenizer_sha256'])!=protocol['tokenizer_sha256']):
        raise ValueError('Vocabulary measurements belong to a different AE/tokenizer')
    metrics = saved['metrics'][:,:2].copy()
    if len(metrics)!=len(units):
        raise ValueError('Vocabulary measurement ordering differs')
    scores = dict(zip(units,metrics))
    allowed = {w for w,u in vocab.items() if len(u)==1 or
               (len(u)<=32 and (not valid_utf8(u) or (scores[u][0] and scores[u][1]<=threshold)))}
    reachable = {w for w,u in vocab.items() if len(u)==1}
    merges = []
    for merge in spec['model']['merges']:
        a,b = merge.split(' ') if isinstance(merge,str) else merge
        if a in reachable and b in reachable and a+b in allowed:
            merges.append(merge); reachable.add(a+b)
    spec = deepcopy(spec)
    spec['normalizer']=None; spec['added_tokens']=[]; spec['post_processor']=None
    spec['model']['vocab']={w:v for w,v in spec['model']['vocab'].items() if w in reachable}
    spec['model']['merges']=merges; spec['model']['ignore_merges']=False
    Tokenizer.from_str(json.dumps(spec)).save(str(out/'tokenizer.json'))
    vectors,_ = encoder.encode(units)
    np.save(out/'vectors.npy',vectors,allow_pickle=False)
    np.save(out/'quality.npy',metrics,allow_pickle=False)
    (out/'units.hex').write_text('\n'.join(u.hex() for u in units)+'\n',encoding='ascii')
    meta = dict(protocol=protocol,fingerprint=fingerprint(protocol),static_strings=len(units),
                reachable_tokens=len(reachable),files={name:sha256(out/name) for name in
                ('tokenizer.json','vectors.npy','quality.npy','units.hex')})
    write_json(out/'manifest.json',meta)
    return meta


class BPEPacker:
    def __init__(self, assets, encoder, capacity=100000):
        assets=Path(assets)
        self.manifest=json.loads((assets/'manifest.json').read_text(encoding='utf8'))
        self.threshold=self.manifest['protocol']['threshold']
        self.tokenizer=Tokenizer.from_file(str(assets/'tokenizer.json'))
        inverse=byte_inverse()
        self.token_bytes={v:bytes(inverse[c] for c in k) for k,v in self.tokenizer.get_vocab().items()}
        units=[bytes.fromhex(v) for v in (assets/'units.hex').read_text().splitlines()]
        vectors=np.load(assets/'vectors.npy',mmap_mode='r')
        quality=np.load(assets/'quality.npy',mmap_mode='r')
        self.static={u:(bool(q[0]),float(q[1]),z) for u,q,z in zip(units,quality,vectors)}
        self.encoder=encoder; self.capacity=int(capacity)
        self.scores=OrderedDict(); self.results=OrderedDict(); self.stats=Counter()

    def ok(self, value):
        return value[0] and value[1]<=self.threshold

    def ensure(self, units):
        found={}; missing=[]
        for u in dict.fromkeys(units):
            if u in self.static: found[u]=self.static[u]
            elif u in self.scores:
                found[u]=self.scores[u]; self.scores.move_to_end(u)
            elif not 1<=len(u)<=32 or not valid_utf8(u):
                found[u]=(False,float('inf'),None)
            else: missing.append(u)
        if missing:
            vectors,quality=self.encoder.encode(missing,score=True)
            self.stats['ae_scored']+=len(missing)
            for u,q,z in zip(missing,quality,vectors):
                value=(bool(q[0]),float(q[1]),z.copy())
                found[u]=value
                if self.capacity:
                    self.scores[u]=value
                    if len(self.scores)>self.capacity:self.scores.popitem(last=False)
        return found

    def pack(self, texts):
        bases=[]
        for text,encoded in zip(texts,self.tokenizer.encode_batch(texts,add_special_tokens=False)):
            parts=[]; buffer=b''
            for i in encoded.ids:
                buffer+=self.token_bytes[i]
                if valid_utf8(buffer):
                    parts.extend([buffer] if len(buffer)<=32 else [c.encode() for c in buffer.decode()])
                    buffer=b''
            if buffer or b''.join(parts)!=text.encode():raise ValueError('BPE changed raw bytes')
            bases.append(parts)
        values=self.ensure(u for row in bases for u in row)
        chars=self.ensure(c.encode() for row in bases for u in row if not self.ok(values[u]) for c in u.decode())
        values.update(chars)
        corrected=[]
        for row in bases:
            parts=[]
            for u in row:
                if self.ok(values[u]):parts.append(u)
                else:
                    for c in u.decode():
                        raw=c.encode()
                        if not values[raw][0]:raise ValueError('Unreadable UTF-8 fallback character: '+raw.hex())
                        self.stats['character_quality_exception']+=int(not self.ok(values[raw]))
                        parts.append(raw)
            corrected.append(parts)
        groups=[[tuple(row[i:i+3]) for i in range(0,len(row),3)] for row in corrected]
        pending={}; ready={}
        for row in groups:
            for key in row:
                self.stats['group_requests']+=1
                if key in self.results:
                    ready[key]=self.results[key]; self.results.move_to_end(key)
                    self.stats['result_hits']+=1
                elif key not in pending:pending[key]=None
        full=self.ensure(b''.join(k) for k in pending); values.update(full)
        pairs=self.ensure(p for k in pending if len(k)==3 and not self.ok(values[b''.join(k)])
                          for p in (k[0]+k[1],k[1]+k[2])); values.update(pairs)
        for key in pending:
            joined=b''.join(key)
            selected=(joined,)
            if not self.ok(values[joined]):
                choices=[]
                if len(key)==3:
                    for label,parts in [('12_3',(key[0]+key[1],key[2])),('1_23',(key[0],key[1]+key[2]))]:
                        if all(self.ok(values[p]) for p in parts):
                            choices.append((sum(values[p][1] for p in parts),label,parts))
                selected=min(choices)[2] if choices else key
            result=(selected,np.stack([values[u][2] for u in selected]))
            ready[key]=result
            if self.capacity:
                self.results[key]=result
                if len(self.results)>self.capacity:self.results.popitem(last=False)
        output=[]
        for text,row in zip(texts,groups):
            units=[u for key in row for u in ready[key][0]]
            vectors=np.concatenate([ready[key][1] for key in row]) if row else np.empty((0,48),np.float16)
            if b''.join(units)!=text.encode():raise ValueError('Packed units changed raw bytes')
            output.append((units,vectors))
        return output
