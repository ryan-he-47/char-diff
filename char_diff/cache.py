from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import time
import urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
from .utils import ROOT,sha256,write_json

DATASET="roneneldan/TinyStories"
REVISION="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FILES={"train":["train-00000-of-00004-2d5a1467fff1081b.parquet",
                "train-00001-of-00004-5852b56a2bd28fd9.parquet",
                "train-00002-of-00004-a26307300439e943.parquet",
                "train-00003-of-00004-d243063613e5a057.parquet"],
       "validation":["validation-00000-of-00001-869c898b519ad725.parquet"]}

def download(url,path,guard):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        return path
    pending=path.with_name(path.name+".pending")
    req=urllib.request.Request(url,headers={"User-Agent":"char-diff/0.1"})
    start=last=time.monotonic();received=0
    with urllib.request.urlopen(req,timeout=60) as r,open(pending,"wb") as w:
        length=int(r.headers.get("Content-Length",0))
        guard.check(length or 300*1024**2)
        while True:
            b=r.read(4*1024**2)
            if not b:break
            w.write(b);received+=len(b)
            if time.monotonic()-last>10:
                print(f"download {path.name}: {received/1e6:.1f} MB, {received/max(1,time.monotonic()-start)/1e6:.1f} MB/s",flush=True)
                last=time.monotonic();guard.check(8*1024**2)
        w.flush();os.fsync(w.fileno())
    if length and received!=length:raise RuntimeError("Incomplete dataset download")
    os.replace(pending,path)
    return path

def raw_rows(split,download_dir,guard,jsonl=None):
    if jsonl:
        with open(jsonl,encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]
        return
    import pyarrow.parquet as pq
    for name in FILES[split]:
        path=download(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}",
                      Path(download_dir)/name,guard)
        try:
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=1024,columns=["text"]):
                    yield from batch.column(0).to_pylist()
        finally:
            # Also close/remove this task's shard when a row limit ends iteration.
            # The completed cache then contains only compact indexed assets.
            path.unlink(missing_ok=True)

def build_windows(root,split,window=512,block=16,carry=128):
    offsets=np.fromfile(Path(root)/f"{split}.offsets.u64",dtype="<u8")
    segments=[];ends=[0];valid=[]
    def save(parts):
        segments.extend(parts);ends.append(len(segments));valid.append(sum(p[2]-p[1] for p in parts))
    # Bounded packing buffers keep memory use independent of corpus size.
    seg_path=Path(root)/f"{split}.segments.i64"
    end_path=Path(root)/f"{split}.windows.u64"
    valid_path=Path(root)/f"{split}.valid.u32"
    total_segments=0;total_windows=0
    with open(seg_path,"wb") as sf,open(end_path,"wb") as ef,open(valid_path,"wb") as vf:
        np.array([0],dtype="<u8").tofile(ef)
        for base in range(0,len(offsets)-1,2048):
            short=[]
            for doc in range(base,min(base+2048,len(offsets)-1)):
                n=int(offsets[doc+1]-offsets[doc])
                if n>window:
                    for st in range(0,n,window):save([(doc,st,min(st+window,n),max(0,st-carry))])
                else:short.append((doc,0,n,0))
            bins=[];free=[]
            for part in sorted(short,key=lambda p:p[2],reverse=True):
                slots=((part[2]+block-1)//block)*block
                candidates=[i for i,v in enumerate(free) if v>=slots]
                if candidates:
                    j=min(candidates,key=lambda i:free[i]);bins[j].append(part);free[j]-=slots
                else:bins.append([part]);free.append(window-slots)
            for parts in bins:save(parts)
            if segments:
                np.asarray(segments,dtype="<i8").tofile(sf)
                (np.asarray(ends[1:],dtype="<u8")+total_segments).tofile(ef)
                np.asarray(valid,dtype="<u4").tofile(vf)
                total_segments+=len(segments);total_windows+=len(valid)
            segments.clear();ends[:]=[0];valid.clear()
            if base%100352==0:print(f"pack {split}: {base}/{len(offsets)-1} stories",flush=True)
    return total_windows

def prepare(out,codec,guard,train_jsonl=None,validation_jsonl=None,limit_train=0,limit_validation=0,
            window=512,block=16,carry=128):
    out=Path(out).resolve();out.relative_to(guard.root)
    if block < 1 or window < block or window % block or carry < 0 or limit_train < 0 or limit_validation < 0:
        raise ValueError("Require a positive block-aligned window, nonnegative carry and row limits")
    if out.exists():raise FileExistsError(f"Cache path already exists: {out}")
    stage=out.with_name(out.name+".building");stage.mkdir(parents=True,exist_ok=False)
    guard.check(512*1024**2)
    db=sqlite3.connect(stage/"dedup.sqlite")
    db.execute("PRAGMA journal_mode=OFF");db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE validation(h BLOB PRIMARY KEY) WITHOUT ROWID")
    db.execute("CREATE TABLE training(h BLOB PRIMARY KEY) WITHOUT ROWID")
    stats=Counter();validation=[]
    for i,text in enumerate(raw_rows("validation",stage/"download",guard,validation_jsonl)):
        if limit_validation and i>=limit_validation:break
        if not text:stats["validation_empty"]+=1;continue
        digest=hashlib.sha256(text.encode()).digest()
        if db.execute("INSERT OR IGNORE INTO validation VALUES (?)",(digest,)).rowcount:
            validation.append(("dev" if digest[0]<128 else "test",text))
        else:stats["validation_duplicates"]+=1
    db.commit()
    unit_ids={};counts=[];unit_bytes=0
    handles={s:(open(stage/f"{s}.ids.u32","wb"),open(stage/f"{s}.offsets.u64","wb")) for s in ("train","dev","test")}
    totals={s:0 for s in handles};docs={s:0 for s in handles};raw_bytes={s:0 for s in handles}
    for _,h in handles.values():np.array([0],dtype="<u8").tofile(h)
    vb=open(stage/"vectors.f32","wb");ub=open(stage/"units.bin","wb");uo=open(stage/"unit_offsets.u64","wb")
    np.array([0],dtype="<u8").tofile(uo)
    def commit(texts,split):
        nonlocal unit_bytes
        units=codec.split(texts)
        unique=list(dict.fromkeys(u for doc in units for u in doc if u not in unit_ids))
        if unique:
            vectors=codec.encode(unique)
            vectors.astype("<f4").tofile(vb)
            for u in unique:
                unit_ids[u]=len(unit_ids);counts.append(0);ub.write(u);unit_bytes+=len(u)
                np.array([unit_bytes],dtype="<u8").tofile(uo)
        flat=[];offsets=[]
        for text,us in zip(texts,units):
            ids=[unit_ids[u] for u in us];flat.extend(ids);totals[split]+=len(ids);offsets.append(totals[split]);docs[split]+=1
            raw_bytes[split]+=len(text.encode())
            if split=="train":
                for ident in ids:counts[ident]+=1
        np.asarray(flat,dtype="<u4").tofile(handles[split][0]);np.asarray(offsets,dtype="<u8").tofile(handles[split][1])
    batch=[];start=last=time.monotonic()
    for i,text in enumerate(raw_rows("train",stage/"download",guard,train_jsonl)):
        if limit_train and i>=limit_train:break
        stats["train_rows_seen"]+=1
        if not text:stats["train_empty"]+=1;continue
        digest=hashlib.sha256(text.encode()).digest()
        if db.execute("SELECT 1 FROM validation WHERE h=?",(digest,)).fetchone():stats["train_validation_overlap"]+=1;continue
        if not db.execute("INSERT OR IGNORE INTO training VALUES (?)",(digest,)).rowcount:stats["train_duplicates"]+=1;continue
        batch.append(text)
        if len(batch)>=128:
            commit(batch,"train");batch.clear();db.commit()
            if time.monotonic()-last>10:
                print(f"prepare: {docs['train']} stories, {totals['train']/1e6:.2f} M units, {len(unit_ids)} unique, {docs['train']/max(1,time.monotonic()-start):.1f} stories/s",flush=True)
                guard.check(32*1024**2);last=time.monotonic()
    if batch:commit(batch,"train")
    for split in ("dev","test"):
        texts=[text for group,text in validation if group==split]
        for i in range(0,len(texts),128):commit(texts[i:i+128],split)
    for pair in handles.values():
        for f in pair:f.flush();os.fsync(f.fileno());f.close()
    for f in [vb,ub,uo]:f.flush();os.fsync(f.fileno());f.close()
    db.close();(stage/"dedup.sqlite").unlink()
    np.asarray(counts,dtype="<u8").tofile(stage/"train_counts.u64")
    if totals["train"]==0:raise RuntimeError("Training split is empty")
    vectors=np.memmap(stage/"vectors.f32",dtype="<f4",mode="r",shape=(len(unit_ids),48))
    cnt=np.asarray(counts,dtype=np.float64)
    mean=np.zeros(48,np.float64);second=0.0
    for i in range(0,len(vectors),8192):
        z=np.asarray(vectors[i:i+8192],dtype=np.float64);c=cnt[i:i+8192]
        mean+=(z*c[:,None]).sum(0);second+=(z*z*c[:,None]).sum()
    mean/=totals["train"];scale=float(np.sqrt(max((second/totals["train"]-mean@mean)/48,1e-12)))
    del vectors
    windows={s:build_windows(stage,s,window,block,carry) for s in handles}
    manifest={"format":1,"dataset":DATASET,"revision":REVISION,"sample_cache":bool(train_jsonl or validation_jsonl or limit_train or limit_validation),
              "sources":{s:{"path":str(p),"sha256":sha256(p)} for s,p in (("train_jsonl",train_jsonl),("validation_jsonl",validation_jsonl)) if p},
              "codec_runtime":{"device":str(codec.device),"precision":"fp32"},
              "unit_policy":{"segment":"word","l_max":32,"merge":"prev","entropy_bits":0.2,"floor_bytes":3},
              "codec":codec.fingerprint,"normalizer":{"mean":mean.tolist(),"scale":scale},
              "layout":{"window":window,"block":block,"carry":carry},"unique_units":len(unit_ids),
              "splits":{s:{"stories":docs[s],"units":totals[s],"raw_bytes":raw_bytes[s],"windows":windows[s]} for s in handles},
              "cleaning":dict(stats),"verify_resplit":{"flagged_unique":codec.verifier.n_flag,"floor_residual":codec.verifier.n_floor_fail},
              "files":{p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in stage.iterdir() if p.is_file()},
              "elapsed_seconds":time.monotonic()-start}
    write_json(stage/"manifest.json",manifest);guard.check()
    os.replace(stage,out)
    print(json.dumps({"cache":str(out),"splits":manifest["splits"],"unique":len(unit_ids)},ensure_ascii=False),flush=True)
    return manifest
