"""Archive step-comparison metrics and inspect endpoint/codec-cycle distances."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from char_diff.codec import Codec
from char_diff.utils import ROOT,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",nargs="?",default="runs/sampling_steps_715121")
    args=p.parse_args();folder=ROOT/args.run;torch.set_num_threads(4)
    report=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    rows=[json.loads(v) for v in (folder/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows)!=96:raise ValueError("The planned 96 outputs are required")
    vectors=np.load(folder/"latents.npz")
    manifest=json.loads((ROOT/"data/tinystories/manifest.json").read_text(encoding="utf-8"))
    mean=np.asarray(manifest["normalizer"]["mean"],np.float32);scale=manifest["normalizer"]["scale"]
    codec=Codec(device="cuda");cycle={};distances={}
    for steps in [16,32,64]:
        subset=[r for r in rows if r["steps"]==steps];zs=[];units=[]
        for r in subset:
            zs.append(vectors[f"doc_{r['doc']}_seed_{r['seed']}_steps_{steps}"])
            raw=bytes.fromhex(r["generated_hex"]);ends=np.r_[0,np.cumsum(r["unit_lengths"])]
            if ends[-1]!=len(raw):raise ValueError("Byte length mismatch")
            units.extend(raw[int(a):int(b)] for a,b in zip(ends[:-1],ends[1:]))
        zs=np.concatenate(zs);valid=np.array([0<len(v)<=codec.ae.l_max for v in units])
        values=[v for v,ok in zip(units,valid) if ok]
        encoded=codec.encode(values);repeated=[]
        for start in range(0,len(encoded),512):repeated.extend(codec.decode(encoded[start:start+512]))
        errors=np.sqrt(np.mean((((encoded-mean)/scale)-zs[valid])**2,axis=1))
        cycle[str(steps)]={"units":len(units),"reencodable":int(valid.sum()),
                           "byte_roundtrip_exact":sum(a==b for a,b in zip(values,repeated)),
                           "mean_rmse":float(errors.mean()),"median_rmse":float(np.median(errors)),
                           "p90_rmse":float(np.quantile(errors,.9))}
        print(json.dumps({"steps":steps,"cycle":cycle[str(steps)]}),flush=True)
    for low,high in [(16,32),(32,64)]:
        first=[];whole=[]
        for r in rows:
            if r["steps"]!=low:continue
            a=vectors[f"doc_{r['doc']}_seed_{r['seed']}_steps_{low}"]
            b=vectors[f"doc_{r['doc']}_seed_{r['seed']}_steps_{high}"]
            first.append(float(np.sqrt(np.mean((a[:16]-b[:16])**2))))
            whole.append(float(np.sqrt(np.mean((a-b)**2))))
        distances[f"{low}_vs_{high}"]={"first_block_mean":float(np.mean(first)),"first_block_median":float(np.median(first)),
                                        "whole_output_mean":float(np.mean(whole))}
    report["codec_cycle"]=cycle;report["endpoint_distances"]=distances
    report["timing_ratio_vs32"]={k:v["timing_seconds_mean"]/report["groups"]["32"]["timing_seconds_mean"] for k,v in report["groups"].items()}
    write_json(folder/"analysis.json",report)
    write_json(ROOT/"docs/experiments/sampling-steps/evidence/sampling-steps-715121-summary.json",report)
    shutil.copyfile(folder/"samples.jsonl",ROOT/"docs/experiments/sampling-steps/evidence/sampling-steps-715121-samples.jsonl")
    shutil.copyfile(folder/"samples.md",ROOT/"docs/experiments/sampling-steps/samples.md")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.style":"normal"})
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    steps=[16,32,64];rates=[100*report["groups"][str(v)]["seen_training_unit_rate"] for v in steps]
    axes[0].plot(steps,rates,marker="o",color="#185d91",lw=2)
    for x,y in zip(steps,rates):axes[0].annotate(f"{y:.2f}%",(x,y),xytext=(0,10),textcoords="offset points",ha="center")
    axes[0].set_ylim(min(90,min(rates)-1),100);axes[0].set_ylabel("Exact matches to training byte units (%)");axes[0].set_title("32 fixed continuations per step count")
    seconds=[report["groups"][str(v)]["timing_seconds_mean"] for v in steps]
    axes[1].bar(steps,seconds,width=9,color="#d58836")
    for x,y in zip(steps,seconds):axes[1].text(x,y+.25,f"{y:.2f}s",ha="center")
    axes[1].set_ylim(0,max(seconds)*1.2);axes[1].set_ylabel("Seconds / 128 generated units");axes[1].set_title("Same four timing prompts, seed 42")
    for ax in axes:
        ax.set_xlabel("Euler steps per 16-unit block");ax.set_xticks(steps);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
        for side in ["top","right"]:ax.spines[side].set_visible(False)
    fig.savefig(ROOT/"docs/experiments/sampling-steps/evidence/sampling-steps-715121.png",dpi=180);plt.close(fig)
    print(json.dumps({"groups":report["groups"],"paired_vs32":report["paired_vs32"],"endpoint_distances":distances}),flush=True)


if __name__=="__main__":main()
