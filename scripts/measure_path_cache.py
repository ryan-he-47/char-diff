"""Measure compact segmentation, lossless vector storage and quantization readout."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import zstandard as zstd
from char_diff.utils import sha256, write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
from scripts.analyze_path_latents import encode, edit_distance


@torch.inference_mode()
def readout(ae, z, units, device, batch_size=512):
    hits, edits, invalid = 0, 0, 0
    for start in range(0, len(units), batch_size):
        part = units[start:start+batch_size]
        zz = torch.as_tensor(z[start:start+len(part)], dtype=torch.float32, device=device)
        k = ae.l_max + 1
        logits = ae.decoder(zz.repeat_interleave(k, 0), torch.arange(k, device=device).repeat(len(part))).view(len(part), k, 257)
        for target, pred in zip(part, logits.argmax(-1).cpu().tolist()):
            hits += pred[:len(target)+1] == list(target)+[256]
            output = bytes(pred[:pred.index(256)] if 256 in pred else pred)
            edits += edit_distance(target, output)
            try: output.decode('utf-8')
            except UnicodeDecodeError: invalid += 1
    return {'units':len(units),'exact':hits,'em':hits/len(units),
            'byte_edit_rate':edits/sum(map(len,units)), 'invalid_utf8':invalid}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    docs=[json.loads(line) for line in a.paths.read_text(encoding='utf8').splitlines()]
    if any(not r['coverage'] for r in docs): raise RuntimeError('Pilot contains uncovered documents; inspect path coverage first')
    units=[];raws=[];byte_offsets=[0];unit_offsets=[0];languages=[]
    language_names=sorted({r['language'] for r in docs})
    for r in docs:
        raw=bytes.fromhex(r['source_hex']);us=[bytes.fromhex(u['hex']) for u in r['units']]
        assert b''.join(us)==raw
        for u in us: u.decode('utf8');assert 1<=len(u)<=32
        units.extend(us);raws.append(raw);byte_offsets.append(byte_offsets[-1]+len(raw));unit_offsets.append(len(units))
        languages.append(language_names.index(r['language']))
    raw=b''.join(raws);lengths=np.array([len(u) for u in units],dtype='u1')
    metadata=np.asarray(byte_offsets,dtype='<u8').tobytes()+np.asarray(unit_offsets,dtype='<u8').tobytes()+bytes(languages)
    bit_values=((lengths[:,None]-1)>>np.arange(5))&1
    packed_lengths=np.packbits(bit_values.reshape(-1),bitorder='little').tobytes()
    restored=np.unpackbits(np.frombuffer(packed_lengths,dtype='u1'),bitorder='little')[:len(units)*5].reshape(-1,5)@(1<<np.arange(5))+1
    assert np.array_equal(restored,lengths)
    compressor=zstd.ZstdCompressor(level=3);decompressor=zstd.ZstdDecompressor()
    payloads={}
    def save(name,blob):
        compressed=compressor.compress(blob)
        assert decompressor.decompress(compressed)==blob
        (a.out/name).write_bytes(blob);(a.out/(name+'.zst')).write_bytes(compressed)
        payloads[name]={'bytes':len(blob),'zstd3_bytes':len(compressed),'ratio_to_text':len(blob)/len(raw),
                        'zstd3_ratio_to_text':len(compressed)/len(raw)}
    save('text.bin',raw);save('lengths.u8',lengths.tobytes());save('lengths.packed5',packed_lengths);save('doc_index.bin',metadata)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(saved['cfg'])).to(a.device).eval();ae.load_state_dict(saved['model']);ae.requires_grad_(False)
    tick=time.perf_counter();z=encode(ae,units,a.device,512);torch.cuda.synchronize()
    encode_seconds=time.perf_counter()-tick
    unique=list(dict.fromkeys(units));lookup={u:i for i,u in enumerate(unique)};ids=np.array([lookup[u] for u in units],dtype='<u4')
    first={}
    for i,u in enumerate(units): first.setdefault(u,i)
    scale=float(np.sqrt(np.mean((z-z.mean(0))**2)))
    fp16=z.astype('<f2');bf16=torch.from_numpy(z).to(torch.bfloat16)
    qscale=(np.maximum(np.max(np.abs(z),axis=1),1e-8)/127).astype('<f2')
    q=np.clip(np.rint(z/qscale.astype('f4')[:,None]),-127,127).astype('i1')
    schemes={'fp32':z,'fp16':fp16.astype('f4'),'bf16':bf16.float().numpy(),
             'int8_row_scale':q.astype('f4')*qscale.astype('f4')[:,None]}
    save('vectors.f32',z.astype('<f4').tobytes());save('vectors.f16',fp16.tobytes())
    save('vectors.bf16',bf16.view(torch.uint16).numpy().astype('<u2').tobytes())
    save('vectors.int8',q.tobytes()+qscale.tobytes())
    for name,values,width in [('f32',z.astype('<f4'),4),('f16',fp16,2)]:
        blob=values.tobytes();shuffled=np.frombuffer(blob,dtype='u1').reshape(-1,width).T.copy().tobytes()
        assert np.frombuffer(shuffled,dtype='u1').reshape(width,-1).T.copy().tobytes()==blob
        save(f'vectors.{name}.shuffle',shuffled)
    unique_z=fp16[[first[u] for u in unique]]
    save('dedup_vectors.f16',unique_z.tobytes());save('unit_ids.u32',ids.tobytes())
    quality={}
    for name,zz in schemes.items():
        quality[name]={**readout(ae,zz,units,a.device),
                       'normalized_rmse':float(np.sqrt(np.mean(((zz-z)/scale)**2))),
                       'max_abs_raw_error':float(np.max(np.abs(zz-z)))}
    sizes={
        'text_u8_boundaries':len(raw)+len(lengths)+len(metadata),
        'text_packed5_boundaries':len(raw)+len(packed_lengths)+len(metadata),
        'fp32_vectors_lengths_index':z.nbytes+len(lengths)+len(metadata),
        'fp16_vectors_lengths_index':fp16.nbytes+len(lengths)+len(metadata),
        'int8_vectors_scales_lengths_index':q.nbytes+qscale.nbytes+len(lengths)+len(metadata),
        'dedup_fp16_ids_lengths_index':unique_z.nbytes+ids.nbytes+len(lengths)+len(metadata)}
    result={'protocol':{'checkpoint_sha256':sha256(a.checkpoint),'paths_sha256':sha256(a.paths),'script_sha256':sha256(__file__),
                       'zstd_version':zstd.__version__,'compression':'zstd level3, each array separately, whole pilot as one shard',
                       'index':'two (docs+1) uint64 arrays for byte/unit offsets, one language byte per document',
                       'lengths':'1..32 bytes, uint8 or L-1 packed in 5 bits; UTF8-safe boundaries inherited',
                       'quantization':'raw AE coordinates; row-wise int8 scale stored fp16, no clipping beyond row absmax',
                       'role':'local calibration pilot, all selected units re-decoded including true EOS; no official training-cache change'},
            'documents':len(docs),'source_bytes':len(raw),'units':len(units),'bytes_per_unit':len(raw)/len(units),
            'unique_units':len(unique),'unique_ratio':len(unique)/len(units),'length_histogram':dict(Counter(map(int,lengths))),
            'encode_seconds':encode_seconds,'encode_units_per_second':len(units)/encode_seconds,
            'payloads':payloads,'total_uncompressed_bytes':sizes,'quantization_quality':quality}
    combos={'text_u8_boundaries':['text.bin','lengths.u8','doc_index.bin'],
            'text_packed5_boundaries':['text.bin','lengths.packed5','doc_index.bin'],
            'fp16_vectors_lengths_index':['vectors.f16','lengths.u8','doc_index.bin'],
            'fp16_shuffle_vectors_lengths_index':['vectors.f16.shuffle','lengths.u8','doc_index.bin'],
            'fp32_shuffle_vectors_lengths_index':['vectors.f32.shuffle','lengths.u8','doc_index.bin']}
    result['total_zstd3_bytes']={k:sum(payloads[n]['zstd3_bytes'] for n in names) for k,names in combos.items()}
    write_json(a.out/'results.json',result)
    print(json.dumps({k:result[k] for k in ['documents','source_bytes','units','bytes_per_unit','unique_ratio','encode_seconds','total_uncompressed_bytes','total_zstd3_bytes','quantization_quality']}),flush=True)


if __name__=='__main__':main()
