"""Segment an approved, ordered UTF-8 JSONL source into resumable permanent shards.

Input rows contain text and language. Corpus selection, deduplication and heldout
exclusion belong to the upstream manifest. This entrypoint retains every row.
Only one writer may own an output directory at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from char_diff.ae_path_runtime import CompiledLazyGraph,FrequentScoreCache
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.coarse_paths import solve_coarse_batch
from char_diff.segmented_cache import fingerprint,read_segmented_shard,write_segmented_shard
from char_diff.utils import sha256,write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE

LANGUAGES=('english','spanish','chinese','turkish','arabic','vietnamese',
           'azerbaijani','korean','german','farsi','hindi','hebrew')


def read_group(handle, max_documents, max_bytes):
    rows=[];size=0
    while len(rows)<max_documents:
        start=handle.tell();line=handle.readline()
        if not line:break
        if not line.strip():continue
        row=json.loads(line);raw=row['text'].encode('utf8')
        if not raw:raise ValueError(f'Empty source document at offset {start}')
        if rows and size+len(raw)>max_bytes:
            handle.seek(start);break
        digest=hashlib.sha256(raw).hexdigest()
        if row.get('doc_id') and row['doc_id']!=digest:raise ValueError(f'Document hash mismatch at {start}')
        if row['language'] not in LANGUAGES:raise ValueError(f'Unknown language at {start}')
        rows.append({'raw':raw,'language':row['language'],'doc_id':digest})
        size+=len(raw)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--proposal',choices=['grid8','bpe','bpe8'],default='grid8')
    p.add_argument('--tokenizer',type=Path,default=Path('assets/tokenizer.json'))
    p.add_argument('--solver',choices=['lazy','eager'],default='eager')
    p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--cache-capacity',type=int,default=500000)
    p.add_argument('--group-documents',type=int,default=64)
    p.add_argument('--group-bytes',type=int,default=262144)
    p.add_argument('--shard-bytes',type=int,default=8*1024**2)
    p.add_argument('--stop-after-shards',type=int,default=0,
                   help='Stop after this many additional atomic shard commits (0: finish source)')
    a=p.parse_args()
    if min(a.batch_size,a.group_documents,a.group_bytes,a.shard_bytes)<1 or a.stop_after_shards<0:
        p.error('Batch, group and shard sizes must be positive; stop count must be nonnegative')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(__file__).resolve().parents[1]
    code_files=['char_diff/ae_nll.py','char_diff/ae_quality.py','char_diff/ae_paths.py',
                'char_diff/ae_lazy_paths.py','char_diff/ae_path_runtime.py','char_diff/ae_scoring_runtime.py',
                'char_diff/path_proposals.py','char_diff/coarse_paths.py','char_diff/segmented_cache.py',
                'char_diff/vendor/bltz/models/autoencoder.py','char_diff/vendor/bltz/models/encoder.py',
                'char_diff/vendor/bltz/models/head.py','scripts/segment_path_shards.py']
    source_hash=sha256(a.source)
    scope={'checkpoint_sha256':sha256(a.checkpoint),'precision':'fp32','tf32':False,
           'batch_size':a.batch_size,'torch':torch.__version__,'device':torch.cuda.get_device_name(),
           'code':{name:sha256(root/name) for name in code_files}}
    provenance={'source_sha256':source_hash,'source_format':'ordered-jsonl-text-language',
                'languages':list(LANGUAGES),'scoring':scope,'quality':'byte+EOS total NLL',
                'quality_weight':10.,'feasibility':'exact byte+EOS argmax','max_unit_bytes':32,
                'boundaries':'UTF8 codepoints','proposal':a.proposal,'solver':a.solver,
                'tokenizer_sha256':sha256(a.tokenizer) if a.proposal.startswith('bpe') else None,
                'cache_capacity':a.cache_capacity,'group_documents':a.group_documents,
                'group_bytes':a.group_bytes,'shard_bytes':a.shard_bytes}
    a.out.mkdir(parents=True,exist_ok=True);owner=a.out/'owner.json'
    if owner.exists():
        if json.loads(owner.read_text(encoding='utf8'))!=provenance:
            raise ValueError('Existing output belongs to a different source or segmentation configuration')
    else:
        if any(a.out.iterdir()):raise ValueError('Expected an empty output directory')
        write_json(owner,provenance)
    key=fingerprint(provenance);next_offset=0;documents=units=raw_bytes=0;existing=[]
    committed_paths=sorted(path for path in a.out.glob('shard-*') if re.fullmatch(r'shard-\d{6}',path.name))
    for index,path in enumerate(committed_paths):
        if path.name!=f'shard-{index:06d}':raise ValueError('Non-contiguous committed shards')
        meta=json.loads((path/'manifest.json').read_text(encoding='utf8'))
        if meta['fingerprint']!=key or fingerprint(meta['provenance'])!=key:
            raise ValueError('Shard provenance mismatch')
        if meta['metadata']['source_start']!=next_offset:raise ValueError('Source cursor gap')
        next_offset=meta['metadata']['source_end'];documents+=meta['documents']
        units+=meta['units'];raw_bytes+=meta['raw_bytes'];existing.append(path)
    if existing:read_segmented_shard(existing[-1],key)
    cache=FrequentScoreCache(a.cache_capacity,scope);snapshot=a.out/'scores.bin'
    if snapshot.exists():cache.load(snapshot)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    scorer=OptimizedAENLLScorer(ae,device='cuda',batch_size=a.batch_size);scorer([b'warmup'])
    CompiledLazyGraph(b'a').optimistic_path()
    tokenizer=Tokenizer.from_file(str(a.tokenizer)) if a.proposal.startswith('bpe') else None
    started=time.perf_counter();additional=0;committed=len(existing);finished=False
    with a.source.open('rb') as source:
        source.seek(next_offset)
        while not finished:
            shard_start=source.tell();records=[];shard_raw=0;statistics={}
            while shard_raw<a.shard_bytes:
                group_start=source.tell()
                group=read_group(source,a.group_documents,a.group_bytes)
                if not group:finished=True;break
                try:
                    rows,stats=solve_coarse_batch(group,scorer,mode=a.proposal,tokenizer=tokenizer,
                                                score_cache=cache,batch_size=a.batch_size,lazy=a.solver=='lazy')
                except Exception as exc:
                    write_json(a.out/'failure.json',{'source_offset':group_start,'error':str(exc),
                               'doc_ids':[d['doc_id'] for d in group],'committed_shards':committed})
                    raise
                for key2,value in stats.items():statistics[key2]=statistics.get(key2,0)+value
                for row in rows:
                    doc=row['row'];records.append({'raw':doc['raw'],'language':doc['language'],
                                                   'doc_id':doc['doc_id'],'lengths':list(map(len,row['path'].units))})
                    shard_raw+=len(doc['raw'])
            if records:
                meta=write_segmented_shard(a.out/f'shard-{committed:06d}',records,provenance,
                     metadata={'source_start':shard_start,'source_end':source.tell(),'statistics':statistics})
                committed+=1;additional+=1;documents+=meta['documents'];units+=meta['units'];raw_bytes+=meta['raw_bytes']
                next_offset=source.tell()
                if a.cache_capacity:cache.save(snapshot)
                print(json.dumps({'committed_shards':committed,'documents':documents,'units':units,
                                  'raw_bytes':raw_bytes,'statistics':statistics}),flush=True)
            write_json(a.out/'progress.json',{'fingerprint':key,'status':'complete' if finished else 'partial',
                       'committed_shards':committed,'documents':documents,'units':units,'raw_bytes':raw_bytes,
                       'source_offset':next_offset,'cache':cache.statistics(),
                       'invocation_seconds':time.perf_counter()-started})
            if a.stop_after_shards and additional>=a.stop_after_shards:break


if __name__=='__main__':main()
