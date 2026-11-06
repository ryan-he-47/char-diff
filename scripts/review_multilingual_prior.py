"""Common raw-prefix byte EM and story samples for either multilingual prior."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.bpe_packing import FrozenUnits,BPEPacker
from char_diff.coarse_paths import solve_coarse_batch
from char_diff.flow import generate
from char_diff.model import Denoiser
from char_diff.utils import ROOT,sha256,write_json


class Readout:
    def __init__(self,encoder,normalizer,goal):
        self.encoder=encoder; self.mean=torch.tensor(normalizer['mean'],device=encoder.device)
        self.scale=normalizer['scale'];self.goal=goal;self.count=0;self.units=[]

    def update(self,blocks):
        with torch.inference_mode(),torch.autocast(device_type=self.encoder.device.type,enabled=False):
            for block in blocks[self.count:]:
                self.units.extend(bytes(b) for b in self.encoder.scorer.ae.decode(block.float()*self.scale+self.mean))
        self.count=len(blocks)
        return sum(map(len,self.units))>=self.goal

    def schedule(self,blocks):
        self.update(blocks);size=sum(map(len,self.units))
        return 1 if size<128 else 4 if size<512 else 8


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--heldout',type=Path,default=ROOT/'data/multilingual_stories/test.jsonl')
    p.add_argument('--steps',type=int,default=32);p.add_argument('--max-prefixes',type=int,default=0)
    p.add_argument('--prefixes-per-language',type=int,default=8)
    p.add_argument('--phase',choices=['all','teacher','samples'],default='all')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config'];normalizer=saved['normalizer'];state=saved['state']
    model=Denoiser(cfg['model']);model.load_state_dict(saved['ema']);model=model.cuda().eval();del saved
    encoder=FrozenUnits(ROOT/cfg['data']['ae_checkpoint'],'cuda',cfg['data']['ae_batch'])
    packer=BPEPacker(ROOT/cfg['data']['bpe_assets'],encoder,100000) if cfg['data']['mode']=='bpe' else None
    docs=defaultdict(list)
    with a.heldout.open(encoding='utf8') as f:
        for line in f:
            doc=json.loads(line);raw=doc['text'].encode();lang=doc['language']
            if len(docs[lang])<a.prefixes_per_language and len(raw)>=160:docs[lang].append(doc)
    selected=[(lang,i,doc) for lang,rows in sorted(docs.items()) for i,doc in enumerate(rows)]
    if a.max_prefixes:selected=selected[:a.max_prefixes]
    protocol=dict(checkpoint_sha256=sha256(a.checkpoint),heldout_sha256=sha256(a.heldout),mode=cfg['data']['mode'],
        weights='ema',steps=a.steps,seeds=[42,123],prefix_bytes=96,teacher_spans=[4,8,16],
        native_blocks=[1,4,8],schedule='B1 before128B, B4 before512B, B8 thereafter',state=state,
        prefixes_per_language=a.prefixes_per_language,max_prefixes=a.max_prefixes)
    if (a.out/'protocol.json').exists() and json.loads((a.out/'protocol.json').read_text(encoding='utf8'))!=protocol:
        raise ValueError('Evaluation directory uses a different protocol')
    write_json(a.out/'protocol.json',protocol)
    existing={}
    if (a.out/'outputs.jsonl').exists():
        for line in (a.out/'outputs.jsonl').read_text(encoding='utf8').splitlines():
            row=json.loads(line);existing[row['key']]=row
    mean=np.asarray(normalizer['mean'],np.float32);scale=normalizer['scale']
    with (a.out/'outputs.jsonl').open('a',encoding='utf8',buffering=1) as log:
        for lang,index,doc in selected:
            raw=doc['text'].encode();prefix=raw[:96].decode('utf8',errors='ignore').encode();target=raw[len(prefix):]
            if packer:units,z=packer.pack([prefix.decode()])[0]
            else:
                solved,_=solve_coarse_batch([dict(raw=prefix,language=lang)],encoder.scorer,
                                          mode='grid8',weight=10.,batch_size=1024,lazy=False)
                units=solved[0]['path'].units;z=encoder.encode(units)[0]
            prefix_z=(z.astype(np.float32)-mean)/scale
            for phase in ('teacher','samples'):
                if a.phase not in ('all',phase) or (phase=='samples' and index>=2):continue
                for mode in ([1,4,8] if phase=='teacher' else [1,4,8,'schedule']):
                    for seed in (42,123):
                        key=f'{phase}/{lang}/{index}/{mode}/{seed}'
                        if key in existing:continue
                        goal=16 if phase=='teacher' else 1024;max_blocks=32 if phase=='teacher' else 2048
                        readout=Readout(encoder,normalizer,goal)
                        schedule=readout.schedule if mode=='schedule' else lambda blocks,size=mode:size
                        tick=time.perf_counter()
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                            generated=generate(model,prefix_z,units=max_blocks*8,steps=a.steps,seed=seed,
                                block_schedule=schedule,stop_condition=readout.update,max_blocks=max_blocks)
                        torch.cuda.synchronize();seconds=time.perf_counter()-tick
                        output=b''.join(readout.units)
                        try:output.decode('utf8');valid=True
                        except UnicodeDecodeError:valid=False
                        row=dict(key=key,phase=phase,language=lang,seed=seed,block=mode,
                            document_id=doc.get('id',hashlib.sha256(raw).hexdigest()),prefix=prefix.decode(),
                            output=output[:goal].decode('utf8',errors='ignore'),output_hex=output.hex(),
                            valid_utf8=valid,empty_units=sum(not u for u in readout.units),
                            units=len(readout.units),blocks=readout.count,seconds=seconds,
                            reached_bytes=len(output)>=goal,bytes_per_second=len(output)/max(seconds,1e-9))
                        if phase=='teacher':row['em']={str(n):len(output)>=n and output[:n]==target[:n] for n in (4,8,16)}
                        else:
                            eligible=[(i,u) for i,u in enumerate(readout.units) if 0<len(u)<=32]
                            eligible=[(i,u) for i,u in eligible if is_utf8(u)]
                            if eligible:
                                encoded,_=encoder.encode([u for _,u in eligible])
                                recon=(encoded.astype('f4')-mean)/scale
                                original=generated[[i for i,_ in eligible]].float().cpu().numpy()
                                errors=np.sqrt(np.square(recon-original).mean(1))
                                row['cycle_rmse_mean']=float(errors.mean());row['cycle_rmse_p90']=float(np.quantile(errors,.9))
                        log.write(json.dumps(row,ensure_ascii=False)+'\n');existing[key]=row
                        print(json.dumps(dict(key=key,seconds=seconds,units=len(readout.units))),flush=True)
    summary={}
    for row in existing.values():
        if row['phase']!='teacher':continue
        for lang in (row['language'],'all'):
            key=f'{lang}/B{row["block"]}';s=summary.setdefault(key,dict(n=0,hits4=0,hits8=0,hits16=0))
            s['n']+=1
            for n in (4,8,16):s[f'hits{n}']+=int(row['em'][str(n)])
    for s in summary.values():
        for n in (4,8,16):s[f'em{n}']=s[f'hits{n}']/s['n']
    write_json(a.out/'summary.json',summary)


def is_utf8(raw):
    try:raw.decode('utf8');return True
    except UnicodeDecodeError:return False


if __name__=='__main__':main()
