"""Summarize existing path benchmarks; no scoring, downloads or training."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.utils import write_json


if __name__=='__main__':
    root=Path('runs/path_optimization_20261105')
    def read(path):return json.loads(path.read_text(encoding='utf8'))
    refs=[json.loads(s) for s in (root/'reference384.jsonl').read_text(encoding='utf8').splitlines()]
    variants=read(root/'coarse384/results.json')['variants']
    inventory=read(Path('runs/multilingual_prior_build_20261105/full_source/inventory.json'))
    rate=read(root/'v100/scorer_min20/results.json')['variants']['bucketed_factored']['candidates_per_second']
    eager=read(root/'v100_coarse/eager/results.json')['variants']
    lazy=read(root/'v100_coarse/lazy/results.json')['variants']
    old=read(Path('runs/multilingual_prior_build_20261105/full_source/preprocessing_cost_estimate.json'))
    cache=read(root/'optimized384_cache/results.json')['timing']['cache']
    total_bytes=sum(r['source_bytes'] for r in refs)
    total_units=sum(r['k'] for r in refs);total_cost=sum(r['cost'] for r in refs)
    source_bytes=sum(r['arrow_bytes_estimate'] for r in inventory['languages'])
    output={'reference':{'documents':len(refs),'bytes':total_bytes,'units':total_units,'cost':total_cost},
            'source_arrow_bytes_estimate':source_bytes,'full_graph_scoring_gpu_hours_after_refactor':
                old['estimated_lazy_candidates']/rate/3600,
            'full_graph_cache_hit_fraction':cache['hits']/(cache['hits']+cache['misses']),
            'variants':{},'per_language':{}}
    for key,variant in variants.items():
        mode=variant['mode'];unit_estimate=candidate_estimate=weighted_cost=weighted_ref_cost=weighted_ref_units=0.
        lang_stats=[]
        for language in inventory['languages']:
            name=language['language'];base=[r for r in refs if r['language']==name]
            rows=[r for r in variant['rows'] if r['language']==name]
            size=sum(r['source_bytes'] for r in base);scale=language['arrow_bytes_estimate']/size
            n=sum(r['units'] for r in rows);c=sum(r['cost'] for r in rows)
            ref_n=sum(r['k'] for r in base);ref_c=sum(r['cost'] for r in base)
            candidates=sum(r['unique_candidates'] for r in rows)
            unit_estimate+=scale*n;candidate_estimate+=scale*candidates
            weighted_cost+=scale*c;weighted_ref_cost+=scale*ref_c;weighted_ref_units+=scale*ref_n
            lang_stats.append({'language':name,'documents':len(rows),'coverage':sum(r['coverage'] for r in rows),
                               'unit_increase':n/ref_n-1,'cost_increase':c/ref_c-1,
                               'candidate_edges_per_source_byte':candidates/size})
        v100={}
        for solver,data in [('eager',eager[mode]),('lazy',lazy[mode])]:
            c=data['cache']
            v100[solver]={'seconds':data['seconds'],'scored_candidates':data['scored_candidates'],
                          'same_paths':data['same_paths'],'cache_hit_fraction':c['hits']/(c['hits']+c['misses']),
                          'wall_hours_direct_byte_extrapolation':data['seconds']*source_bytes/total_bytes/3600}
        output['variants'][mode]={'units':variant['units'],'unit_increase':variant['units']/total_units-1,
                                 'cost_increase':variant['cost']/total_cost-1,'bytes_per_unit':variant['bytes_per_unit'],
                                 'estimated_source_units':unit_estimate,'model_D_over_20':unit_estimate/20,
                                 'weighted_unit_increase':unit_estimate/weighted_ref_units-1,
                                 'weighted_cost_increase':weighted_cost/weighted_ref_cost-1,
                                 'estimated_eager_candidates_before_global_cache':candidate_estimate,
                                 'eager_scoring_hours_at_measured_long_candidate_rate':candidate_estimate/rate/3600,
                                 'v100':v100}
        output['per_language'][mode]=lang_stats
    output['interpretation']={
        'candidate_estimates':'32 documents/language; weighted by upstream Arrow bytes; before full-corpus dedup and persistent-cache savings',
        'wall_estimates':'Direct source-size extrapolation from 384-story V100 end-to-end run; sample language mixture; excludes corpus IO, permanent shard compression, queue and selected-vector encoding',
        'cost':'K + 10 * sum(byte+EOS NLL); this is the path objective, downstream generation is evaluated separately',
        'choice':'grid8/eager for first full preparation; full BPE retained as a higher-cost, lower-path-cost alternative; BPE8 retained for diagnostics'}
    write_json(root/'analysis.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ('per_language','interpretation')},indent=2))
