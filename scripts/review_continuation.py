"""Summarize the paired continuation logs and run their common fixed dev evaluation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.data import LatentCache
from char_diff.model import Denoiser
from char_diff.utils import ROOT,write_json
from scripts.review_checkpoint import denoising


@torch.inference_mode()
def sample_panel(folder,arms=("uniform","beta21")):
    from char_diff.codec import Codec
    from char_diff.flow import generate
    selected=json.loads((ROOT/"docs/experiments/baseline/evidence/main-run-715121-prompts.json").read_text(encoding="utf-8"))
    previous=[json.loads(v) for v in (ROOT/"docs/experiments/history-reencoding/evidence/history-reencoding-715121-pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    controls={(r["doc"],r["seed"]):r for r in previous if r["units"]==128}
    cache=LatentCache(ROOT/"data/tinystories","dev");codec=Codec(device="cuda");models={}
    mean=torch.tensor(cache.mean,device="cuda");scale=cache.scale
    kinds={}
    for arm in arms:
        state=torch.load(folder/arm/"last.pt",map_location="cpu",weights_only=False)
        if state["cache_fingerprint"]!=cache.fingerprint or state["codec"]!=codec.fingerprint:raise ValueError("Sampling representations differ")
        model=Denoiser(state["config"]["model"]).cuda().eval();model.load_state_dict(state["ema"]);models[arm]=model
        kinds[arm]=state["config"]["train"].get("prediction_type","x0")
        del state
    offsets=np.fromfile(cache.root/"unit_offsets.u64",dtype="<u8");blob=(cache.root/"units.bin").read_bytes();counts=np.fromfile(cache.root/"train_counts.u64",dtype="<u8")
    seen={blob[int(offsets[i]):int(offsets[i+1])] for i in np.flatnonzero(counts)}
    rows=[];vectors={}
    if (folder/"samples.jsonl").exists():raise FileExistsError("Sampling output already exists")
    with open(folder/"samples.jsonl","x",encoding="utf-8",buffering=1) as log:
        for prompt in selected["prompts"]:
            for seed in [42,123]:
                doc=prompt["doc"];base=controls[(doc,seed)]
                if base["prompt"]!=prompt["prompt"]:raise ValueError("Cached prompt differs")
                for arm in ["parent",*arms]:
                    if arm=="parent":
                        record=dict(base["arms"]["continuous"]);source="validated_cached_parent"
                    else:
                        with torch.autocast("cuda",dtype=torch.float16):z=generate(models[arm],cache.story(doc)[:32],128,32,seed,prediction_type=kinds[arm])
                        parts=codec.decode(z*scale+mean);raw=b"".join(parts)
                        try:text=raw.decode("utf-8");utf8=True
                        except UnicodeDecodeError:text=raw.decode("utf-8",errors="replace");utf8=False
                        record={"generated":text,"generated_hex":raw.hex(),"unit_lengths":[len(v) for v in parts],
                                "utf8_valid":utf8,"seen_training_units":sum(v in seen for v in parts),"empty_units":sum(not v for v in parts)}
                        vectors[f"{arm}_doc_{doc}_seed_{seed}"]=z.cpu().numpy();source="fresh"
                    record.update(arm=arm,doc=doc,seed=seed,units=128,prompt=prompt["prompt"],source=source)
                    rows.append(record);log.write(json.dumps(record,ensure_ascii=False)+"\n")
                    print(json.dumps({"arm":arm,"doc":doc,"seed":seed,"seen_training_units":record["seen_training_units"],"utf8_valid":record["utf8_valid"]}),flush=True)
    np.savez(folder/"sample_latents.npz",**vectors)
    groups={arm:{"cases":sum(r["arm"]==arm for r in rows),"units":4096,"seen_training_units":sum(r["seen_training_units"] for r in rows if r["arm"]==arm),
                 "utf8_valid":sum(r["utf8_valid"] for r in rows if r["arm"]==arm)} for arm in ["parent",*arms]}
    mapping={(r["doc"],r["seed"],r["arm"]):r for r in rows};paired={}
    for arm in arms:
        differences=[mapping[(q["doc"],s,arm)]["seen_training_units"]-mapping[(q["doc"],s,"parent")]["seen_training_units"] for q in selected["prompts"] for s in [42,123]]
        paired[arm]={"more":sum(v>0 for v in differences),"same":sum(v==0 for v in differences),"less":sum(v<0 for v in differences)}
    write_json(folder/"sample-summary.json",{"protocol":{"prefix_units":32,"units":128,"steps":32,"weights":"ema","history":"raw","seeds":[42,123],"prompts":16,"precision":"fp16","codec_precision":"fp32","prediction_types":kinds},"groups":groups,"paired_vs_parent":paired})
    lines=["# 原始x0与两个实验臂：固定前缀逐组输出",""]
    for q in selected["prompts"]:
        for seed in [42,123]:
            lines.extend([f"## doc={q['doc']} / seed={seed}","","**前缀**","",q["prompt"],""])
            for arm in ["parent",*arms]:lines.extend([f"**{arm}**","",mapping[(q['doc'],seed,arm)]["generated"],""])
    (folder/"samples.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"groups":groups,"paired":paired}),flush=True)


def training_summary(folder):
    rows=[json.loads(v) for v in (folder/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    successful=[v for v in rows if v["updated"]]
    def group(values):
        losses=np.array([v["loss"] for v in values]);units=np.array([v["target_units"] for v in values])
        return {"updates":len(values),"mean_loss":float(losses.mean()),"unit_weighted_loss":float((losses*units).sum()/units.sum()),
                "median_grad_norm":float(np.median([v["grad_norm"] for v in values]))}
    result={"updates":len(successful),"attempts":len(rows),"skipped_updates":len(rows)-len(successful),
            "seen_units":rows[-1]["seen_units"],"skipped_units":rows[-1]["skipped_units"],
            "peak_lr":max(v["lr"] for v in rows),"last_lr":rows[-1]["lr"],
            "first_100":group(successful[:100]),"last_100":group(successful[-100:]),
            "first_500":group(successful[:500]),"last_500":group(successful[-500:]),
            "source":json.loads((folder/"lineage.json").read_text(encoding="utf-8")) if (folder/"lineage.json").exists() else None,
            "result":json.loads((folder/"result.json").read_text(encoding="utf-8"))}
    result["last_vs_first500_percent"]=100*(result["last_500"]["unit_weighted_loss"]/result["first_500"]["unit_weighted_loss"]-1)
    return result


@torch.inference_mode()
def main(arms=("uniform","beta21"),default_folder="runs/continuation_review_20261101",evidence_root="docs/experiments/high-noise-continuation"):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--folder",default=default_folder)
    p.add_argument("--samples-only",action="store_true")
    args=p.parse_args();torch.set_num_threads(4);folder=ROOT/args.folder
    if args.samples_only:
        sample_panel(folder,arms);return
    prior=json.loads((ROOT/"docs/experiments/baseline/evidence/main-run-715121-dev-evaluation.json").read_text(encoding="utf-8"))
    parent=prior["stages"][0]
    report={"protocol":{"date":"2026-11-01","split":"dev","windows":64,"units_per_bucket":23501,
                        "times":[.05,.25,.5,.75,.95],"seed":31415,"precision":"fp32","device":torch.cuda.get_device_name(0),
                        "torch":torch.__version__,"baseline_metrics":"reused same fixed evaluation from original review"},
            "training":{},"validation":{"parent":{k:v["full_history"] for k,v in parent["weights"].items()}}}
    for arm in arms:
        local=folder/arm;report["training"][arm]=training_summary(local)
        print(json.dumps({"arm":arm,"training":report["training"][arm]}),flush=True)
        state=torch.load(local/"last.pt",map_location="cpu",weights_only=False)
        state.pop("optimizer",None)
        if state.get("lineage") and state["lineage"]["source_sha256"]!=parent["sha256"]:raise ValueError("Continuation source differs")
        kind=state["config"]["train"].get("prediction_type","x0")
        report["training"][arm]["prediction_type"]=kind
        report["training"][arm]["loss_space"]=state["config"]["train"].get("loss_space","x0")
        cache=LatentCache(ROOT/state["config"]["data"]["cache"],"dev")
        if cache.fingerprint!=state["cache_fingerprint"] or cache.fingerprint!=prior["cache_fingerprint"]:raise ValueError("Evaluation representations differ")
        model=Denoiser(state["config"]["model"]).cuda().eval();report["validation"][arm]={}
        for weights in ["ema","model"]:
            model.load_state_dict(state[weights])
            report["validation"][arm][weights]=denoising(model,cache,64,torch.device("cuda"),prediction_type=kind)
            write_json(folder/"evaluation-progress.json",report)
        del model,state,cache
    write_json(folder/"summary.json",report)
    write_json(ROOT/evidence_root/"evidence/review-20261101.json",report)
    for arm in arms:
        for kind in ["ema","model"]:
            values=report["validation"][arm][kind];base=report["validation"]["parent"][kind]
            print(json.dumps({"arm":arm,"weights":kind,"mse":[v["mse"] for v in values],"relative_percent_vs_parent":[100*(v["mse"]/b["mse"]-1) for v,b in zip(values,base)]}),flush=True)


if __name__=="__main__":main()
