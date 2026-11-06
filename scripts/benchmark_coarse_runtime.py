"""Compare bounded cached coarse solvers against recorded FP32 coarse paths."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from char_diff.ae_path_runtime import CompiledLazyGraph,FrequentScoreCache
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.coarse_paths import solve_coarse_batch
from char_diff.utils import sha256,write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=Path('assets/tokenizer.json'))
    p.add_argument('--modes',nargs='+',default=['grid8','bpe','bpe8'])
    p.add_argument('--solver',choices=['lazy','eager'],default='lazy')
    p.add_argument('--group-documents',type=int,default=64)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    scorer=OptimizedAENLLScorer(ae,device='cuda',batch_size=2048);scorer([b'warmup'])
    CompiledLazyGraph(b'a').optimistic_path()
    tokenizer=Tokenizer.from_file(str(a.tokenizer))
    references=json.loads(a.reference.read_text(encoding='utf8'))
    evidence={'checkpoint_sha256':sha256(a.checkpoint),'reference_sha256':sha256(a.reference),
              'tokenizer_sha256':sha256(a.tokenizer),'gpu':torch.cuda.get_device_name(),
              'solver':a.solver,'group_documents':a.group_documents,'variants':{}}
    for mode in a.modes:
        source=references['variants'][mode[4:] if mode.startswith('grid') else mode]['rows']
        documents=[{'raw':bytes.fromhex(r['source_hex']),'language':r['language'],'doc_id':r['doc_id']} for r in source]
        cache=FrequentScoreCache(500000,{'benchmark':mode});totals={};records=[];started=time.perf_counter()
        for start in range(0,len(documents),a.group_documents):
            rows,stats=solve_coarse_batch(documents[start:start+a.group_documents],scorer,mode=mode,
                                        tokenizer=tokenizer,score_cache=cache,lazy=a.solver=='lazy')
            for key,value in stats.items():totals[key]=totals.get(key,0)+value
            for row in rows:
                ref=source[start+row['index']];path=row['path']
                records.append({'doc_id':ref['doc_id'],'language':ref['language'],'units':len(path.units),
                                'cost':path.cost,'same_path':list(path.boundaries)==ref['boundaries'],
                                'cost_difference':path.cost-ref['cost'],'source_bytes':len(row['row']['raw']),
                                'rounds':row['rounds']})
        data={'seconds':time.perf_counter()-started,**totals,'documents':len(records),
              'same_paths':sum(r['same_path'] for r in records),'units':sum(r['units'] for r in records),
              'max_abs_cost_difference':max(abs(r['cost_difference']) for r in records),
              'cache':cache.statistics(),'rows':records}
        evidence['variants'][mode]=data;write_json(a.out/'results.json',evidence)
        print(json.dumps({mode:{k:v for k,v in data.items() if k not in ('rows','cache')}}),flush=True)
