"""Roundtrip actual NLL10 documents and exercise bounded cache eviction/rebuild."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.segmented_cache import write_segmented_shard,read_segmented_shard,FP16ShardCache,fingerprint


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];source=root/'runs/multilingual_prior_build_20261105'
    paths=[json.loads(s) for s in (source/'paths/paths.jsonl').read_text(encoding='utf8').splitlines()]
    records=[{'raw':bytes.fromhex(r['source_hex']),'lengths':[u['bytes'] for u in r['units']],
              'language':r['language'],'doc_id':r['doc_id']} for r in paths]
    provenance={'languages':sorted({r['language'] for r in records}),'ae_sha256':'26df106186699b7ef13a0d1c4db096e107a701d7fb4e71c2621d99c3e5d6ea61',
                'lambda':10,'precision':'fp32','purpose':'real-pilot storage check'}
    z=np.fromfile(source/'cache_measurement/vectors.f16',dtype='<f2').reshape(-1,48)
    with tempfile.TemporaryDirectory(prefix='char_diff_shard_') as temporary:
        base=Path(temporary);meta=write_segmented_shard(base/'shard',records,provenance)
        loaded=read_segmented_shard(base/'shard',fingerprint(provenance))
        for i,row in enumerate(records):
            left,right=map(int,loaded['byte_offsets'][i:i+2]);lo,hi=map(int,loaded['unit_offsets'][i:i+2])
            assert loaded['text'][left:right]==row['raw']
            assert loaded['lengths'][lo:hi].tolist()==row['lengths']
        assert loaded['manifest']['units']==len(z)
        calls={};cache=FP16ShardCache(base/'hot',provenance,2*500*48*2)
        def builder(key):
            def build(path):
                calls[key]=calls.get(key,0)+1;z[key*500:(key+1)*500].tofile(path)
            return build
        for key in [0,1,2,0]:
            indices=np.array([4,0,499]);result=cache.read(str(key),500,indices,builder(key))
            assert np.array_equal(result,z[key*500:(key+1)*500][indices].astype('f4'))
            assert sum(p.stat().st_size for p in cache._files())<=cache.limit
        assert calls=={0:2,1:1,2:1},calls
        assert not list(cache.root.glob('*.pending-*'))
        try:read_segmented_shard(base/'shard','different-source')
        except ValueError:pass
        else:raise AssertionError('Expected provenance mismatch')
        evidence={'documents':len(records),'units':len(z),'raw_bytes':meta['raw_bytes'],
                  'stored_payload_bytes':sum(f['stored_bytes'] for f in meta['files'].values()),
                  'exact_document_and_length_roundtrip':True,'eviction_and_rebuild_calls':calls,
                  'gathered_fp16_values_exact':True,'provenance_mismatch_rejected':True}
    (source/'segmented_cache_check.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence))
