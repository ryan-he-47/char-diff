"""Pinned Parquet ingestion, disk deduplication and atomic corpus segmentation.

One writer owns the output. A committed shard precedes the SQLite cursor commit;
recovery adopts such a shard if an interruption occurred between these steps.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.request import urlopen

import pyarrow.parquet as pq

from .ae_path_runtime import FrequentScoreCache
from .coarse_paths import solve_coarse_batch
from .segmented_cache import fingerprint,read_segmented_shard,write_segmented_shard
from .utils import sha256,write_json


def owned_unlink(path, parent):
    path=Path(path)
    if path.resolve().parent!=Path(parent).resolve() or path.is_symlink():
        raise ValueError('Unexpected source-cache path')
    path.unlink(missing_ok=True)


def fetch_source(root, source, entry):
    if 'fixture_path' in entry:return Path(entry['fixture_path']),False
    if 'local_path' in entry:
        path = Path(entry['local_path'])
        if sha256(path) != entry['sha256']:
            raise ValueError('Materialized source checksum mismatch')
        return path, False
    cache=Path(root)/'input_cache';cache.mkdir(exist_ok=True)
    filename=entry['language']+'_'+Path(entry['path']).name
    path=cache/filename
    if path.exists():
        if path.stat().st_size==entry['bytes'] and sha256(path)==entry['sha256']:return path,True
        raise ValueError('Existing source cache failed its checksum')
    url=f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{entry['path']}?download=true"
    temporary=path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        digest=hashlib.sha256();size=0
        try:
            with urlopen(url,timeout=90) as response,temporary.open('wb') as output:
                while block:=response.read(8*1024**2):
                    output.write(block);digest.update(block);size+=len(block)
            if size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
                raise ValueError('Downloaded source checksum mismatch')
            temporary.replace(path);return path,True
        except Exception:
            owned_unlink(temporary,cache)
            if attempt==2:raise
            time.sleep(2*(attempt+1))


class CorpusIndex:
    def __init__(self,root,source,holdout_hashes):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.source=source
        claim={'source':source,'old_holdout_sha256':hashlib.sha256(holdout_hashes).hexdigest(),
               'rule':'upstream train; exclude upstream test and old calibration/test; global exact-byte dedup'}
        owner=self.root/'source.json'
        if owner.exists():
            if json.loads(owner.read_text(encoding='utf8'))!=claim:raise ValueError('Source ownership mismatch')
        else:
            if any(self.root.iterdir()):raise ValueError('Expected empty corpus output')
            write_json(owner,claim)
        self.connection=sqlite3.connect(self.root/'index.sqlite')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('PRAGMA cache_size=-2097152')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS excluded (digest BLOB PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        ''')
        if self.get('state') is None:
            counts={language:{k:0 for k in ('scanned','empty','excluded','duplicate','documents','raw_bytes','units')}
                    for language in source['languages']}
            self.set('state',{'cursor':[0,0],'shards':0,'counts':counts,'complete':False})
            self.connection.commit()
        self.holdout_hashes=holdout_hashes

    def get(self,key):
        row=self.connection.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self,key,value):
        self.connection.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value)))

    def prepare_exclusions(self,on_file=None):
        if len(self.holdout_hashes)%32:raise ValueError('Malformed old heldout hash file')
        done=self.get('excluded_files') or 0
        self.connection.executemany('INSERT OR IGNORE INTO excluded VALUES (?)',
             ((self.holdout_hashes[i:i+32],) for i in range(0,len(self.holdout_hashes),32)))
        self.connection.commit()
        for i,entry in enumerate(self.source['test_files'][done:],done):
            path,owned=fetch_source(self.root,self.source,entry)
            parquet=pq.ParquetFile(path)
            try:
                for batch in parquet.iter_batches(batch_size=4096,columns=['story']):
                    rows=[]
                    for value in batch.column(0).to_pylist():
                        if not isinstance(value,str):raise ValueError('Expected string test story')
                        if value:rows.append((hashlib.sha256(value.encode('utf8')).digest(),))
                    self.connection.executemany('INSERT OR IGNORE INTO excluded VALUES (?)',rows)
                self.set('excluded_files',i+1);self.connection.commit()
            except BaseException:
                self.connection.rollback();raise
            finally:parquet.close()
            if owned:owned_unlink(path,self.root/'input_cache')
            if on_file:on_file(i+1,entry)
        self.set('prepare_complete',True);self.connection.commit()
        return self.connection.execute('SELECT count(*) FROM excluded').fetchone()[0]

    def accept(self,raw):
        digest=hashlib.sha256(raw).digest()
        if self.connection.execute('SELECT 1 FROM excluded WHERE digest=?',(digest,)).fetchone():return 'excluded'
        inserted=self.connection.execute('INSERT OR IGNORE INTO seen VALUES (?)',(digest,)).rowcount
        return 'accepted' if inserted else 'duplicate'

    def close(self):self.connection.close()


def source_records(index,cursor):
    for file_index in range(cursor[0],len(index.source['train_files'])):
        entry=index.source['train_files'][file_index];path,owned=fetch_source(index.root,index.source,entry)
        parquet=pq.ParquetFile(path);skip=cursor[1] if file_index==cursor[0] else 0
        if not 0<=skip<=parquet.metadata.num_rows:raise ValueError('Source cursor beyond Parquet rows')
        offset=0
        try:
            for group in range(parquet.metadata.num_row_groups):
                group_size=parquet.metadata.row_group(group).num_rows
                if offset+group_size<=skip:offset+=group_size;continue
                for batch in parquet.iter_batches(batch_size=4096,row_groups=[group],columns=['story']):
                    for text in batch.column(0).to_pylist():
                        row=offset;offset+=1
                        if row<skip:continue
                        if not isinstance(text,str):raise ValueError(f'Non-string story at {file_index}:{row}')
                        yield {'raw':text.encode('utf8'),'language':entry['language'],
                               'source_file':file_index,'source_row':row},[file_index,row+1]
        finally:parquet.close()
        if owned:owned_unlink(path,index.root/'input_cache')


def semantic_protocol(provenance):
    execution={'group_documents','group_bytes','shard_bytes','cache_capacity','pipeline_workers','cpu_processes'}
    result={k:v for k,v in provenance.items() if k not in execution and k!='scoring'}
    result['scoring']={k:v for k,v in provenance['scoring'].items() if k not in ('code','batch_size','device','cuda_streams')}
    return result


def adopt_shards(index,provenance,accepted_fingerprints=()):
    accepted=set(accepted_fingerprints)|{fingerprint(provenance)}
    def read(path):
        shard=read_segmented_shard(path)
        if shard['manifest']['fingerprint'] not in accepted:raise ValueError('Unrecognized shard runtime generation')
        return shard
    state=index.get('state');folder=index.root/'segments';folder.mkdir(exist_ok=True)
    while (path:=folder/f"shard-{state['shards']:06d}").exists():
        shard=read(path);meta=shard['manifest']['metadata']
        if meta['before_cursor']!=state['cursor'] or meta['after_state']['shards']!=state['shards']+1:
            raise ValueError('Orphan shard cursor mismatch')
        ids=shard['doc_ids']
        index.connection.executemany('INSERT OR IGNORE INTO seen VALUES (?)',
                                    ((ids[i:i+32],) for i in range(0,len(ids),32)))
        state=meta['after_state'];index.set('state',state);index.connection.commit()
    if state['shards']:
        read(folder/f"shard-{state['shards']-1:06d}")
    return state


def segment_corpus(index,scorer,provenance,*,batch_size=2048,cache_capacity=500000,
                   group_documents=64,group_bytes=262144,shard_bytes=8*1024**2,
                   max_shards=0,on_commit=None,on_group=None,guard=None,should_stop=None,after_publish=None,
                   pipeline_workers=0,upgrade_runtime=False,process_solver=None):
    if not index.get('prepare_complete'):raise ValueError('Heldout preparation is incomplete')
    owner=index.root/'segmentation.json'
    accepted=set();history=index.root/'runtime_history'
    for path in sorted(history.glob('*.json')):
        transition=json.loads(path.read_text(encoding='utf8'))
        for generation in ('old','new'):
            if semantic_protocol(transition[generation])!=semantic_protocol(provenance):
                raise ValueError('Runtime history changes corpus semantics')
            accepted.add(fingerprint(transition[generation]))
    if owner.exists():
        previous=json.loads(owner.read_text(encoding='utf8'))
        if previous!=provenance:
            if not upgrade_runtime or semantic_protocol(previous)!=semantic_protocol(provenance):
                raise ValueError('Segmentation ownership mismatch')
            current=adopt_shards(index,previous,accepted)
            transition={'old':previous,'new':provenance,'at_state':current,
                        'reason':'User-authorized throughput optimization; graph/AE/FP32/NLL remain fixed'}
            history.mkdir(exist_ok=True)
            write_json(history/(fingerprint(previous)[:16]+'-'+fingerprint(provenance)[:16]+'.json'),transition)
            accepted.add(fingerprint(previous));write_json(owner,provenance)
    else:write_json(owner,provenance)
    state=adopt_shards(index,provenance,accepted)
    if state['complete']:return state
    if process_solver is not None:
        from .compiled_score_cache import CompiledScoreCache
        cache=CompiledScoreCache(cache_capacity,provenance['scoring'])
    else:cache=FrequentScoreCache(cache_capacity,provenance['scoring'])
    snapshot=index.root/('scores-'+fingerprint(provenance['scoring'])[:16]+'.bin')
    use_pipeline=bool(pipeline_workers or process_solver is not None)
    if not use_pipeline and not upgrade_runtime and not history.exists() and (index.root/'scores.bin').exists():
        snapshot=index.root/'scores.bin'
    if snapshot.exists():cache.load(snapshot)
    source=iter(source_records(index,state['cursor']));additional=0;ended=False
    try:
        while not ended:
            if guard:guard()
            before=copy.deepcopy(state);records=[];record_bytes=0;stats={};tick=time.perf_counter()
            index.connection.execute('BEGIN')
            while record_bytes<shard_bytes:
                group=[];size=0
                target_bytes=shard_bytes-record_bytes if use_pipeline else group_bytes
                while size<target_bytes and (use_pipeline or len(group)<group_documents):
                    try:row,cursor=next(source)
                    except StopIteration:
                        ended=True;state['cursor']=[len(index.source['train_files']),0];break
                    state['cursor']=cursor;counts=state['counts'][row['language']];counts['scanned']+=1
                    if not row['raw']:counts['empty']+=1;continue
                    outcome=index.accept(row['raw'])
                    if outcome!='accepted':counts[outcome]+=1;continue
                    group.append(row);size+=len(row['raw'])
                if group:
                    try:
                        if process_solver is not None:
                            result,timing=process_solver.solve(group,scorer,score_cache=cache,
                                      group_documents=group_documents,group_bytes=group_bytes)
                        elif pipeline_workers:
                            from .coarse_pipeline import solve_coarse_pipeline
                            result,timing=solve_coarse_pipeline(group,scorer,weight=10.,score_cache=cache,
                                      workers=pipeline_workers,group_documents=group_documents,group_bytes=group_bytes)
                        else:
                            result,timing=solve_coarse_batch(group,scorer,mode='grid8',weight=10.,
                                                             score_cache=cache,batch_size=batch_size,lazy=False)
                    except Exception as exc:
                        write_json(index.root/'failure.json',{'error':str(exc),'resume_state':before,
                         'group':[{'source_file':r['source_file'],'source_row':r['source_row'],
                                   'doc_id':hashlib.sha256(r['raw']).hexdigest(),'language':r['language']} for r in group]})
                        raise
                    for key,value in timing.items():stats[key]=stats.get(key,0)+value
                    for item in result:
                        row=item['row'];record={k:v for k,v in row.items() if k!='offsets'}
                        record['lengths']=list(map(len,item['path'].units));records.append(record)
                        record_bytes+=len(row['raw']);counts=state['counts'][row['language']]
                        counts['documents']+=1;counts['raw_bytes']+=len(row['raw']);counts['units']+=len(record['lengths'])
                    if on_group:on_group(state,stats)
                if ended:break
            state['complete']=ended
            if records:
                state['shards']+=1
                write_segmented_shard(index.root/'segments'/f"shard-{state['shards']-1:06d}",records,provenance,
                     metadata={'before_cursor':before['cursor'],'after_state':copy.deepcopy(state),'statistics':stats})
                if after_publish:after_publish()
            index.set('state',state);index.connection.commit()
            if records:
                additional+=1
                if cache_capacity:cache.save(snapshot)
            event={'state':state,'statistics':stats,'wall_seconds':time.perf_counter()-tick,'cache':cache.statistics()}
            write_json(index.root/'progress.json',event)
            if on_commit:on_commit(event)
            if (max_shards and additional>=max_shards) or (should_stop and should_stop()):break
    except BaseException:
        index.connection.rollback();raise
    finally:source.close()
    return state
