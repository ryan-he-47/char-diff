"""Global spectra and scale-dependent nearest-neighbor geometry of frozen byte latents."""
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.utils import ROOT,sha256,write_json


def spectrum(x,weights=None):
    x=np.asarray(x,np.float64)
    w=np.ones(len(x),np.float64) if weights is None else np.asarray(weights,np.float64)
    w=w/w.sum();mean=(x*w[:,None]).sum(0);center=x-mean
    eigen=np.maximum(np.linalg.eigvalsh((center*w[:,None]).T@center),0)[::-1]
    p=eigen/eigen.sum();positive=p[p>0]
    return {"participation_ratio":float(1/(p*p).sum()),"entropy_rank":float(np.exp(-(positive*np.log(positive)).sum())),
            "dimensions_for_variance":{str(q):int(np.searchsorted(p.cumsum(),q)+1) for q in [.9,.95,.99]},
            "eigenvalues":eigen.tolist()}


def nearest(x,queries,references,k=40,device="cuda"):
    data=torch.as_tensor(x,dtype=torch.float32,device=device)
    ref_ids=np.asarray(references);locations={int(v):i for i,v in enumerate(ref_ids)}
    ref=data[torch.as_tensor(ref_ids,device=device)];output=[]
    for start in range(0,len(queries),128):
        ids=np.asarray(queries[start:start+128]);q=data[torch.as_tensor(ids,device=device)]
        # Direct subtraction avoids cancellation for very close lexical variants.
        distances=torch.cdist(q,ref,compute_mode="donot_use_mm_for_euclid_dist")
        self_at=torch.tensor([locations[int(v)] for v in ids],device=device)
        distances[torch.arange(len(ids),device=device),self_at]=float("inf")
        output.append(distances.topk(k,largest=False,sorted=True).values.cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def local_metrics(distances):
    valid=(distances[:,0]>0)&np.isfinite(distances).all(1)
    d=distances[valid]
    result={"queries":len(distances),"used":len(d),"excluded_zero_or_nonfinite":int((~valid).sum()),
            "twonn_mle":float(1/np.log(d[:,1]/d[:,0]).mean()),"mle":{},"neighbor_radius":{}}
    for k in [10,20,40]:
        sums=np.log(d[:,k-1,None]/d[:,:k-1]).sum(1)
        local=(k-1)/sums
        result["mle"][str(k)]={"pooled":float((k-1)/sums.mean()),"local_mean":float(local.mean()),
                                "local_median":float(np.median(local)),"local_p10":float(np.quantile(local,.1)),"local_p90":float(np.quantile(local,.9))}
    for k in [1,2,10,20,40]:
        result["neighbor_radius"][str(k)]={"median":float(np.median(d[:,k-1])),"p10":float(np.quantile(d[:,k-1],.1)),"p90":float(np.quantile(d[:,k-1],.9))}
    return result


def variants(raw,rng,n=16):
    stem=raw.strip();lead=raw[:len(raw)-len(raw.lstrip())];trail=raw[len(raw.rstrip()):]
    found=set();alphabet=b"abcdefghijklmnopqrstuvwxyz"
    for _ in range(1000):
        mode=int(rng.integers(4));i=int(rng.integers(len(stem)));char=bytes([int(rng.choice(list(alphabet)))])
        if mode==0:value=stem[:i]+stem[i+1:]
        elif mode==1:value=stem[:i]+char+stem[i+1:]
        elif mode==2:value=stem[:i]+char+stem[i:]
        else:
            i=min(i,len(stem)-2);value=stem[:i]+stem[i+1:i+2]+stem[i:i+1]+stem[i+2:]
        value=lead+value+trail
        if value!=raw and 0<len(value)<=32:found.add(value)
        if len(found)==n:return sorted(found)
    raise RuntimeError("Could not construct the requested unique edits")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",default="data/tinystories");p.add_argument("--out",default="runs/latent_dimension_20261101")
    p.add_argument("--device",default="cuda");p.add_argument("--queries",type=int,default=1024)
    args=p.parse_args();torch.set_num_threads(4)
    folder=ROOT/args.out;folder.mkdir(parents=True,exist_ok=True)
    cache=ROOT/args.cache;m=json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
    mean=np.asarray(m["normalizer"]["mean"],np.float32);scale=float(m["normalizer"]["scale"])
    vectors=(np.fromfile(cache/"vectors.f32",dtype="<f4").reshape(-1,48)-mean)/scale
    counts=np.fromfile(cache/"train_counts.u64",dtype="<u8");train_ids=np.flatnonzero(counts)
    x=vectors[train_ids];global_stats={"unique_units":spectrum(x),"occurrence_weighted":spectrum(x,counts[train_ids])}
    _,first=np.unique(x,axis=0,return_index=True);first=np.sort(first);point_ids=train_ids[first];x=x[first]
    offsets=np.fromfile(cache/"unit_offsets.u64",dtype="<u8");blob=(cache/"units.bin").read_bytes()
    raw=lambda idx:blob[int(offsets[idx]):int(offsets[idx+1])]
    rng=np.random.default_rng(20261101)
    order=np.argsort(counts[point_ids])[::-1];words=[];stems=set()
    for point in order:
        unit=raw(point_ids[point]);stem=unit.strip()
        if re.fullmatch(rb"[a-z]{4,12}",stem) and stem not in stems:
            words.append(int(point));stems.add(stem)
        if len(words)==2000:break
    bases=rng.choice(words,size=128,replace=False)
    probes=rng.choice(len(x),size=min(args.queries,len(x)),replace=False)
    queries=np.r_[probes,bases];records=[];new_units=set();existing={raw(i) for i in point_ids}
    for idx in bases:
        unit=raw(point_ids[idx]);edits=variants(unit,rng)
        records.append({"unit_id":int(point_ids[idx]),"base_hex":unit.hex(),"variants_hex":[v.hex() for v in edits]})
        new_units.update(v for v in edits if v not in existing)
    known=set(map(int,queries));rest=np.array([i for i in range(len(x)) if i not in known])
    subset=np.r_[np.array(sorted(known)),rng.choice(rest,size=min(10000-len(known),len(rest)),replace=False)]
    groups={};saved={}
    for label,refs in [("natural_10k",subset),("natural_all",np.arange(len(x)))]:
        distances=nearest(x,queries,refs,device=args.device);saved[label]=distances
        groups[label]={"reference_points":len(refs),"random_unit_queries":local_metrics(distances[:len(probes)]),
                       "base_word_queries":local_metrics(distances[len(probes):])}
        print(json.dumps({"condition":label,"twonn":groups[label]["random_unit_queries"]["twonn_mle"],"mle20":groups[label]["random_unit_queries"]["mle"]["20"]["pooled"]}),flush=True)
    codec=Codec(device=args.device);synthetic=sorted(new_units)
    extra=(codec.encode(synthetic)-mean)/scale
    augmented=np.concatenate([x,extra]);_,keep=np.unique(augmented,axis=0,return_index=True)
    # Preserve original point indices while removing exact-coordinate duplicates.
    keep=np.sort(keep);augmented=augmented[keep]
    if not np.array_equal(keep[:len(x)],np.arange(len(x))):raise ValueError("Original query indices changed")
    distances=nearest(augmented,bases,np.arange(len(augmented)),device=args.device)
    saved["with_variants"]=distances
    groups["with_variants"]={"reference_points":len(augmented),"base_word_queries":local_metrics(distances)}
    for dimension in [8,48]:
        reference=rng.normal(size=(10000,dimension)).astype(np.float32)
        if dimension<48:reference=np.pad(reference,((0,0),(0,48-dimension)))
        distances=nearest(reference,np.arange(512),np.arange(len(reference)),device=args.device)
        groups[f"gaussian_{dimension}d"]={"reference_points":10000,"true_dimension":dimension,"queries":local_metrics(distances)}
    report={"date":"2026-11-01","seed":20261101,"cache_fingerprint":sha256(cache/"manifest.json"),"codec":codec.fingerprint,
            "global_spectra":global_stats,"train_unit_types":len(train_ids),"unique_coordinate_points":len(x),
            "base_words":len(bases),"variants_per_word":16,"new_variant_byte_strings":len(synthetic),"groups":groups,
            "protocol":{"metric":"Euclidean in globally scalar-standardized coordinates","self_neighbors":"excluded by point index",
                        "exact_coordinate_duplicates":"removed","knn":"exact direct FP32 pairwise distances; no approximate index",
                        "twonn":"1 / mean(log(r2/r1))","mle_local":"(k-1) / sum_j<k log(rk/rj)",
                        "mle_pooled":"inverse mean reciprocal local dimension","device":str(args.device)},
            "source_sha256":sha256(Path(__file__))}
    np.savez(folder/"neighbor_distances.npz",**saved)
    (folder/"variants.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
    write_json(folder/"summary.json",report)
    write_json(ROOT/"docs/experiments/latent-dimension/evidence/dimension-20261101.json",report)
    print(json.dumps({"spectra":{k:v['participation_ratio'] for k,v in global_stats.items()},"groups":{k:v.get('base_word_queries',v.get('queries')) for k,v in groups.items()}}),flush=True)


if __name__=="__main__":main()
