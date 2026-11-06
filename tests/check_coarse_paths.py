"""Affected UTF-8 proposal, failed-span expansion and coarse solver checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from char_diff.ae_paths import CandidateScore,boundary_offsets,candidate_spans,NoFeasiblePath
from char_diff.ae_path_runtime import FrequentScoreCache
from char_diff.coarse_paths import solve_coarse_batch
from char_diff.path_proposals import proposed_boundaries


if __name__=='__main__':
    tokenizer=Tokenizer.from_file('assets/tokenizer.json')
    texts=['apple applle café cafe\u0301 苹果🙂，你好！'*3,'a'*99,'مرحبا بالعالم، قصة صغيرة. '*3,
           '한국어 이야기 है। Tiếng Việt. '*3]
    docs=[{'raw':s.encode()} for s in texts]
    def scorer(units):
        return [CandidateScore(len(u.decode())==1 or len(u)%7!=0,(sum(u)%5)/100.) for u in units]
    count=0
    for mode in ('grid8','bpe','bpe8'):
        for document in docs:
            raw=document['raw'];nodes=proposed_boundaries(raw,mode,tokenizer=tokenizer)
            assert nodes[0]==0 and nodes[-1]==len(raw) and set(nodes)<=set(boundary_offsets(raw))
            assert max(b-a for a,b in zip(nodes,nodes[1:]))<=32
        for lazy in (False,True):
            cache=FrequentScoreCache(20000,{'test':mode})
            rows,stats=solve_coarse_batch(docs,scorer,mode=mode,tokenizer=tokenizer,score_cache=cache,batch_size=32,lazy=lazy)
            assert stats['fallback_intervals']>0
            for row in rows:
                raw=row['row']['raw'];nodes=set(row['row']['offsets'])
                scores={}
                for a,b in candidate_spans(raw):
                    u=raw[a:b];value=scorer([u])[0]
                    # The full-graph reference uses the same allowed endpoints.
                    if a not in nodes or b not in nodes:value=CandidateScore(False)
                    # These fixtures have repeated strings; endpoint membership
                    # belongs to positions, so compare on an independent byte-DP below.
                    scores[(a,b)]=value
                best={0:(0.,0,-1)};prev={}
                for b in sorted(nodes-{0}):
                    choices=[(best[a][0]+(1+10*scores[a,b].quality),best[a][1]+1,a)
                             for a in nodes if a<b and b-a<=32 and a in best and scores[a,b].feasible]
                    if choices:best[b]=min(choices);prev[b]=best[b][2]
                reverse=[len(raw)]
                while reverse[-1]:reverse.append(prev[reverse[-1]])
                assert row['path'].boundaries==tuple(reversed(reverse))
                assert b''.join(row['path'].units)==raw
                count+=1
    try:solve_coarse_batch([{'raw':b'x'}],lambda units:[CandidateScore(False)]*len(units))
    except NoFeasiblePath:pass
    else:raise AssertionError('Uncoverable source was silently accepted')
    print({'coarse_paths':count,'original_utf8_bytes':True,'failed_span_expansion':True,'uncoverable_raises':True})
