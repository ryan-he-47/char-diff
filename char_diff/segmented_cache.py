"""Permanent UTF-8 segmentation shards and a bounded, regenerable FP16 cache.

The hot cache is owned by one training process; its lock serializes producer
and reader threads. Return gathered copies so eviction never invalidates a batch.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import threading
import uuid

import numpy as np
import zstandard as zstd


def fingerprint(value):
    blob=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def write_segmented_shard(path, records, provenance, *, metadata=None):
    """Atomically write a complete shard. Records contain raw, lengths, language.

    Provenance must bind the source, AE, scorer and path configuration. The
    caller controls shard size and document order; documents remain complete.
    """
    path=Path(path).resolve()
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    staging=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    assert staging.parent==path.parent
    staging.mkdir()
    text=bytearray();lengths=bytearray();byte_offsets=[0];unit_offsets=[0];languages=[];ids=bytearray()
    source_files=[];source_rows=[]
    language_names=provenance['languages']
    try:
        for row in records:
            raw=row['raw'];raw.decode('utf8',errors='strict')
            lens=list(row['lengths'])
            if not raw or not lens or any(type(n) is not int or not 1<=n<=32 for n in lens):
                raise ValueError('Expected a nonempty document and lengths in 1..32')
            if sum(lens)!=len(raw):raise ValueError('Segmentation does not cover the document')
            start=0
            for length in lens:
                raw[start:start+length].decode('utf8',errors='strict');start+=length
            digest=hashlib.sha256(raw).digest()
            if row.get('doc_id') and row['doc_id']!=digest.hex():raise ValueError('Document hash mismatch')
            text.extend(raw);lengths.extend(lens);ids.extend(digest)
            byte_offsets.append(len(text));unit_offsets.append(len(lengths))
            languages.append(language_names.index(row['language']))
            if 'source_file' in row:
                source_files.append(int(row['source_file']));source_rows.append(int(row['source_row']))
        if not languages:raise ValueError('Empty shard')
        arrays={'text.bin':bytes(text),'lengths.u8':bytes(lengths),'doc_ids.sha256':bytes(ids),
                'byte_offsets.u64':np.asarray(byte_offsets,dtype='<u8').tobytes(),
                'unit_offsets.u64':np.asarray(unit_offsets,dtype='<u8').tobytes(),
                'languages.u8':bytes(languages)}
        if source_files:
            if (len(source_files)!=len(languages) or min(source_files)<0 or max(source_files)>65535
                    or min(source_rows)<0):raise ValueError('Invalid per-document source coordinates')
            arrays['source_files.u16']=np.asarray(source_files,dtype='<u2').tobytes()
            arrays['source_rows.u64']=np.asarray(source_rows,dtype='<u8').tobytes()
        compressor=zstd.ZstdCompressor(level=3);files={}
        for name,blob in arrays.items():
            data=compressor.compress(blob);filename=name+'.zst'
            with (staging/filename).open('xb') as handle:
                handle.write(data);handle.flush();os.fsync(handle.fileno())
            files[name]={'file':filename,'raw_bytes':len(blob),'stored_bytes':len(data),
                         'sha256':hashlib.sha256(data).hexdigest()}
        manifest={'format':'utf8-segmented-v1','provenance':provenance,
                  'fingerprint':fingerprint(provenance),'documents':len(languages),
                  'raw_bytes':len(text),'units':len(lengths),'files':files,
                  'metadata':metadata or {}}
        with (staging/'manifest.json').open('x',encoding='utf8',newline='\n') as handle:
            json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.write('\n');handle.flush();os.fsync(handle.fileno())
        if path.exists():raise FileExistsError(path)
        os.replace(staging,path)
        return manifest
    except BaseException:
        # Only direct files created by this invocation belong to this cleanup.
        for child in staging.iterdir():
            if child.is_file() and child.resolve().parent==staging:child.unlink()
        staging.rmdir()
        raise


def read_segmented_shard(path, expected_fingerprint=None):
    path=Path(path);meta=json.loads((path/'manifest.json').read_text(encoding='utf8'))
    if meta['format']!='utf8-segmented-v1':raise ValueError('Unknown shard format')
    if meta['fingerprint']!=fingerprint(meta['provenance']):raise ValueError('Invalid provenance')
    if expected_fingerprint and meta['fingerprint']!=expected_fingerprint:raise ValueError('Shard provenance mismatch')
    decompressor=zstd.ZstdDecompressor();values={}
    for name,info in meta['files'].items():
        target=(path/info['file']).resolve()
        if target.parent!=path.resolve():raise ValueError('Invalid shard member path')
        data=target.read_bytes()
        if hashlib.sha256(data).hexdigest()!=info['sha256']:raise ValueError('Shard checksum mismatch')
        blob=decompressor.decompress(data,max_output_size=info['raw_bytes'])
        if len(blob)!=info['raw_bytes']:raise ValueError('Invalid decompressed length')
        values[name]=blob
    out={'manifest':meta,'text':values['text.bin'],
         'lengths':np.frombuffer(values['lengths.u8'],dtype='u1'),
         'byte_offsets':np.frombuffer(values['byte_offsets.u64'],dtype='<u8'),
         'unit_offsets':np.frombuffer(values['unit_offsets.u64'],dtype='<u8'),
         'languages':np.frombuffer(values['languages.u8'],dtype='u1'),
         'doc_ids':values['doc_ids.sha256']}
    if 'source_files.u16' in values:
        out['source_files']=np.frombuffer(values['source_files.u16'],dtype='<u2')
        out['source_rows']=np.frombuffer(values['source_rows.u64'],dtype='<u8')
        if len(out['source_files'])!=meta['documents'] or len(out['source_rows'])!=meta['documents']:
            raise ValueError('Invalid source coordinate shape')
    if (len(out['text'])!=meta['raw_bytes'] or len(out['lengths'])!=meta['units']
            or len(out['byte_offsets'])!=meta['documents']+1
            or len(out['unit_offsets'])!=meta['documents']+1
            or len(out['languages'])!=meta['documents'] or len(out['doc_ids'])!=32*meta['documents']):
        raise ValueError('Invalid shard dimensions')
    if (out['byte_offsets'][0]!=0 or out['unit_offsets'][0]!=0
            or out['byte_offsets'][-1]!=meta['raw_bytes'] or out['unit_offsets'][-1]!=meta['units']
            or np.any(np.diff(out['byte_offsets'].astype('i8'))<=0)
            or np.any(np.diff(out['unit_offsets'].astype('i8'))<=0)
            or np.any((out['lengths']<1)|(out['lengths']>32))):
        raise ValueError('Invalid shard offsets or unit lengths')
    prefix=np.concatenate((np.array([0],dtype=np.uint64),np.cumsum(out['lengths'],dtype=np.uint64)))
    if not np.array_equal(prefix[out['unit_offsets']],out['byte_offsets']):raise ValueError('Length/offset mismatch')
    return out


class FP16ShardCache:
    """One-process LRU disk cache. Builders write raw little-endian FP16 arrays."""
    def __init__(self, root, provenance, max_bytes, latent_dim=48):
        self.key=fingerprint(provenance);self.root=Path(root).resolve()/self.key
        self.root.mkdir(parents=True,exist_ok=True)
        self.limit=int(max_bytes);self.dim=int(latent_dim);self.lock=threading.RLock()
        if self.limit<1 or self.dim<1:raise ValueError('Invalid cache capacity/dimension')
        owner=self.root/'owner.json'
        value={'format':'fp16-hot-v1','fingerprint':self.key,'latent_dim':self.dim}
        if owner.exists():
            if json.loads(owner.read_text())!=value:raise ValueError('Hot cache ownership mismatch')
        else:owner.write_text(json.dumps(value,indent=2),encoding='utf8')

    def _files(self):
        return [p for p in self.root.glob('*.f16') if p.is_file() and not p.is_symlink()]

    def ensure(self, shard_id, units, builder):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',shard_id):raise ValueError('Unsafe shard identifier')
        expected=int(units)*self.dim*2
        if expected<=0 or expected>self.limit:raise ValueError('Shard exceeds hot-cache capacity')
        with self.lock:
            path=self.root/(shard_id+'.f16')
            if path.is_symlink():raise ValueError('Unexpected symlink in owned cache')
            if path.exists():
                if path.stat().st_size!=expected:raise ValueError('Cached vector shape mismatch')
                os.utime(path,None);return path
            files=sorted(self._files(),key=lambda f:f.stat().st_mtime_ns)
            used=sum(f.stat().st_size for f in files)
            for old in files:
                if used+expected<=self.limit:break
                if old.resolve().parent!=self.root:raise ValueError('Unexpected cache file parent')
                used-=old.stat().st_size;old.unlink()
            temporary=self.root/(shard_id+'.pending-'+uuid.uuid4().hex)
            try:
                builder(temporary)
                if temporary.stat().st_size!=expected:raise ValueError('Builder wrote an incorrect vector shape')
                with temporary.open('r+b') as handle:os.fsync(handle.fileno())
                os.replace(temporary,path)
            finally:
                if temporary.exists():temporary.unlink()
            return path

    def read(self, shard_id, units, indices, builder):
        indices=np.asarray(indices)
        if indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=units):
            raise ValueError('Invalid vector indices')
        with self.lock:
            path=self.ensure(shard_id,units,builder)
            with path.open('rb') as handle, mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ) as view:
                array=np.ndarray((units,self.dim),dtype='<f2',buffer=view)
                selected=array[indices].astype(np.float32,copy=True)
                del array
            return selected
