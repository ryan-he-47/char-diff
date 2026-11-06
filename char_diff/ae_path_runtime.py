"""Compiled DAG relaxation, bounded frequent-fragment scores and GPU scheduling."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import struct
import time

import numpy as np
from numba import njit

from .ae_paths import CandidateScore, NoFeasiblePath, PathResult
from .ae_lazy_paths import LazyNLLGraph


@njit(cache=True, nogil=True)
def _dag_path(costs):
    n, width = costs.shape
    distance=np.full(n,np.inf);count=np.full(n,n,np.int64);previous=np.full(n,-1,np.int64)
    distance[0]=0.;count[0]=0
    for j in range(1,n):
        for k in range(min(j,width)):
            i=j-k-1; value=distance[i]+costs[j,k]; number=count[i]+1
            if (value<distance[j] or (value==distance[j] and math.isfinite(value)
                and (number<count[j] or (number==count[j] and i<previous[j])))):
                distance[j]=value;count[j]=number;previous[j]=i
    if not math.isfinite(distance[-1]):return np.empty(0,np.int64),np.inf
    nodes=np.empty(count[-1]+1,np.int64);j=n-1;at=count[-1]
    while at>=0:
        nodes[at]=j;j=previous[j];at-=1
    return nodes,distance[-1]


class CompiledLazyGraph(LazyNLLGraph):
    def optimistic_path(self):
        nodes,cost=_dag_path(self.costs)
        if not len(nodes):raise NoFeasiblePath(f'No feasible path to byte {len(self.raw)}')
        boundaries=tuple(self.offsets[int(i)] for i in nodes)
        units=tuple(self.raw[a:b] for a,b in zip(boundaries,boundaries[1:]))
        unknown=list(dict.fromkeys(u for u in units if u not in self.known))
        if unknown:return None,unknown
        return PathResult(boundaries,units,float(cost),math.fsum(self.known[u].quality for u in units)),[]


class FrequentScoreCache:
    """Two bounded LRU queues: second use promotes a candidate to the hot queue.

    Both feasible and infeasible outcomes are cached. Entries are exact bytes;
    the caller binds checkpoint, scoring implementation and precision in scope.
    """
    def __init__(self, capacity=250000, scope=None):
        self.capacity=int(capacity)
        if self.capacity<0:raise ValueError('Negative cache capacity')
        self.probation=OrderedDict();self.hot=OrderedDict()
        self.probation_limit=max(1,self.capacity//4)
        self.hot_limit=max(0,self.capacity-self.probation_limit)
        self.scope=scope or {};self.hits=0;self.misses=0;self.evictions=0
        self.length_hits=np.zeros(33,np.int64);self.length_requests=np.zeros(33,np.int64)

    def get(self, unit):
        self.length_requests[len(unit)]+=1
        if unit in self.hot:
            value=self.hot[unit];self.hot.move_to_end(unit)
        elif unit in self.probation:
            value=self.probation.pop(unit)
            if self.hot_limit:
                self.hot[unit]=value
                if len(self.hot)>self.hot_limit:self.hot.popitem(last=False);self.evictions+=1
            else:self.probation[unit]=value
        else:self.misses+=1;return None
        self.hits+=1;self.length_hits[len(unit)]+=1
        return value

    def put(self, unit, value):
        if not self.capacity:return
        if unit in self.hot:self.hot[unit]=value;self.hot.move_to_end(unit);return
        self.probation[unit]=value;self.probation.move_to_end(unit)
        # Let first-use entries occupy unused hot capacity during warmup.
        limit=max(self.probation_limit,self.capacity-len(self.hot))
        while len(self.probation)>limit:self.probation.popitem(last=False);self.evictions+=1

    def statistics(self):
        return {'capacity':self.capacity,'entries':len(self.hot)+len(self.probation),'hot_entries':len(self.hot),
                'hits':self.hits,'misses':self.misses,'evictions':self.evictions,
                'length_hits':self.length_hits.tolist(),'length_requests':self.length_requests.tolist()}

    def save(self, path):
        header=json.dumps({'scope':self.scope,'capacity':self.capacity,'format':1},sort_keys=True).encode()
        blob=bytearray(struct.pack('<I',len(header))+header)
        for tier,queue in [(0,self.probation),(1,self.hot)]:
            for unit,score in queue.items():
                blob.extend(struct.pack('<BBBd',tier,len(unit),score.feasible,score.quality));blob.extend(unit)
        from pathlib import Path
        target=Path(path);temporary=target.with_name(target.name+'.pending')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary.write_bytes(blob);temporary.replace(target)

    def load(self,path):
        from pathlib import Path
        blob=Path(path).read_bytes();size=struct.unpack_from('<I',blob)[0]
        meta=json.loads(blob[4:4+size]);pos=4+size
        if meta['scope']!=self.scope or meta['format']!=1:raise ValueError('Score cache scope mismatch')
        while pos<len(blob):
            tier,length,feasible,q=struct.unpack_from('<BBBd',blob,pos);pos+=11
            unit=blob[pos:pos+length];pos+=length
            if len(unit)!=length or not 1<=length<=32 or tier not in (0,1):raise ValueError('Invalid cache entry')
            queue=self.hot if tier else self.probation;queue[unit]=CandidateScore(bool(feasible),q)
        while len(self.hot)>self.hot_limit:self.hot.popitem(last=False)
        while len(self.probation)>max(0,self.capacity-len(self.hot)):self.probation.popitem(last=False)


def solve_stream(documents, scorer, *, weight=10., active_documents=64, batch_size=2048,
                 score_cache=None, trace=None, on_result=None):
    """One GPU owner overlaps scoring with compiled CPU paths for other docs.

    Finished documents are replaced immediately. A bounded pending queue pools
    duplicate requests; scoring results wake only documents awaiting those keys.
    """
    if min(active_documents,batch_size)<1:raise ValueError('Invalid scheduling capacity')
    cache=score_cache if score_cache is not None else FrequentScoreCache(0)
    _dag_path(np.ones((2,1),np.float64)) # compile outside reported wall time
    iterator=iter(enumerate(documents));states={};ready=deque();pending=OrderedDict();waiters={}
    outputs=[];exhausted=False;future=None;inflight=[];gpu_batches=[]
    timings={'graph_seconds':0.,'path_seconds':0.,'cache_seconds':0.,'score_wall_seconds':0.,'wait_seconds':0.}
    total_candidates=0;inflight_keys=set();started=time.perf_counter()

    def admit():
        nonlocal exhausted
        while not exhausted and len(states)<active_documents:
            try:index,row=next(iterator)
            except StopIteration:exhausted=True;break
            tick=time.perf_counter();raw=row['raw']
            graph=CompiledLazyGraph(raw,weight=weight,offsets=row.get('offsets'))
            initial=row.get('initial_scores')
            if initial:graph.update({u:initial[u] for u in graph.occurrences if u in initial})
            timings['graph_seconds']+=time.perf_counter()-tick
            states[index]={'row':row,'graph':graph,'remaining':set(),'rounds':0}
            ready.append(index)

    def score(units):
        tick=time.perf_counter();values=scorer(units)
        return values,time.perf_counter()-tick

    admit()
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='ae-score') as gpu:
        while states or future is not None:
            if future is not None and future.done():
                values,seconds=future.result();timings['score_wall_seconds']+=seconds
                total_candidates+=len(inflight);gpu_batches.append(len(inflight))
                for unit,value in zip(inflight,values):
                    cache.put(unit,value)
                    if trace is not None:trace(unit,value)
                    for index in waiters.pop(unit,()):
                        state=states[index];state['graph'].update({unit:value});state['remaining'].remove(unit)
                        if not state['remaining']:ready.append(index)
                future=None;inflight=[];inflight_keys.clear()
            if pending and future is None and (len(pending)>=batch_size or not ready):
                inflight=[pending.popitem(last=False)[0] for _ in range(min(batch_size,len(pending)))]
                inflight_keys=set(inflight);future=gpu.submit(score,inflight)
            if ready and len(pending)<2*batch_size:
                index=ready.popleft();state=states[index];tick=time.perf_counter()
                result,unknown=state['graph'].optimistic_path();state['rounds']+=1
                timings['path_seconds']+=time.perf_counter()-tick
                if result is not None:
                    item={'index':index,'row':state['row'],'path':result,'rounds':state['rounds']}
                    outputs.append(item)
                    if on_result is not None:on_result(item)
                    del states[index];admit();continue
                tick=time.perf_counter();hits={};misses=[]
                for unit in unknown:
                    value=cache.get(unit)
                    if value is None:misses.append(unit)
                    else:hits[unit]=value
                state['graph'].update(hits)
                if not misses:ready.append(index)
                else:
                    state['remaining']=set(misses)
                    for unit in misses:
                        waiters.setdefault(unit,set()).add(index)
                        if unit not in inflight_keys:pending[unit]=None
                timings['cache_seconds']+=time.perf_counter()-tick
                continue
            if future is not None:
                tick=time.perf_counter();future.result();timings['wait_seconds']+=time.perf_counter()-tick
            elif states and not ready and not pending:
                raise RuntimeError('Path scheduler stalled')
    return sorted(outputs,key=lambda x:x['index']),{
        'seconds':time.perf_counter()-started,**timings,'scored_candidates':total_candidates,
        'batches':len(gpu_batches),'mean_batch':float(np.mean(gpu_batches)) if gpu_batches else 0.,
        'small_batches':sum(n<batch_size//2 for n in gpu_batches),'cache':cache.statistics()}
