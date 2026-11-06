"""Evaluate the one-pass endpoint on held-out windows and on its full observed alphabet."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
from char_diff.utils import write_json,sha256
from evaluate_utf8_ae import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','calibration','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(args.manifest.read_text(encoding='utf8'))
    by_lang={}
    for part in spec['parts']:
        by_lang.setdefault(part['language'],set()).update(part['characters'])
    args.out.mkdir(parents=True,exist_ok=True)
    inventory=args.out/'character_inventory.jsonl'
    with inventory.open('w',encoding='utf8') as f:
        for lang,chars in sorted(by_lang.items()):
            f.write(json.dumps(dict(split='train',language=lang,id=lang,text=''.join(sorted(chars))),ensure_ascii=False)+'\n')
    torch.backends.cuda.matmul.allow_tf32=False
    evaluate(args.checkpoint,args.calibration,args.out/'heldout.json',train_data=inventory,device='cuda')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['state']['natural_source_bytes']!=spec['raw_bytes']:
        raise ValueError('Incomplete pass at evaluation')
    ae=ByteStringAE(Cfg(saved['cfg']))
    ae.load_state_dict(saved['model'])
    scorer=OptimizedAENLLScorer(ae.eval().requires_grad_(False),device='cuda',batch_size=8192)
    chars=sorted(set().union(*by_lang.values()))
    failures={c for c,s in zip(chars,scorer([c.encode() for c in chars])) if not s.feasible}
    write_json(args.out/'full_coverage.json',dict(checkpoint=sha256(args.checkpoint),manifest=sha256(args.manifest),
               types=len(chars),exact_types=len(chars)-len(failures),failed=sorted(failures),
               languages={k:dict(types=len(v),failed=sorted(v & failures)) for k,v in by_lang.items()}))
if __name__=='__main__':main()
