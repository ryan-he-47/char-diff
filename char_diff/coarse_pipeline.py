"""Dense coarse DAGs and bounded CPU/GPU pipelining for the unchanged grid8 graph."""
from __future__ import annotations
from concurrent.futures import Future,ThreadPoolExecutor
import math
import queue
import threading
import time

import numpy as np

from .ae_path_runtime import _dag_path,FrequentScoreCache
from .ae_paths import boundary_offsets,NoFeasiblePath,PathResult
from .path_proposals import proposed_boundaries


class DenseCoarseGraph:
    def __init__(self,raw,nodes):
        self.raw=raw;self.nodes=tuple(nodes);offsets=np.asarray(nodes,np.int64)
        end=np.arange(1,len(nodes))[:,None];slot=np.arange(32)[None,:]
        previous=end-slot-1
        valid=(previous>=0)&((offsets[end]-offsets[previous.clip(min=0)])<=32)
        rows,slots=np.nonzero(valid);self.ends=rows+1;self.slots=slots
        starts=self.ends-self.slots-1
        self.keys=[raw[nodes[int(a)]:nodes[int(b)]] for a,b in zip(starts,self.ends)]

    def solve(self,scores,weight):
        values=np.fromiter((1.+weight*scores[u].quality if scores[u].feasible else np.inf
                            for u in self.keys),dtype=np.float64,count=len(self.keys))
        costs=np.full((len(self.nodes),32),np.inf)
        costs[self.ends,self.slots]=values
        indices,cost=_dag_path(costs)
        if not len(indices):raise NoFeasiblePath(f'No feasible path to byte {len(self.raw)}')
        nodes=tuple(self.nodes[int(i)] for i in indices)
        units=tuple(self.raw[a:b] for a,b in zip(nodes,nodes[1:]))
        return PathResult(nodes,units,float(cost),math.fsum(scores[u].quality for u in units))


class ScoringService:
    """One GPU/cache owner; independent CPU jobs submit pooled score requests."""
    def __init__(self,scorer,cache):
        self.scorer=scorer;self.cache=cache;self.requests=queue.Queue()
        self.scored=0;self.score_seconds=0.;self.calls=0;self.failed=None
        self.thread=threading.Thread(target=self._run,name='coarse-gpu',daemon=True)
        self.thread.start()

    def score(self,units):
        units=list(dict.fromkeys(units))
        if not units:return {}
        future=Future();self.requests.put((units,future));return future.result()

    def _run(self):
        while True:
            item=self.requests.get()
            if item is None:return
            items=[item]
            # Requests already ready from other CPU jobs share one length pool.
            while len(items)<8:
                try:other=self.requests.get_nowait()
                except queue.Empty:break
                if other is None:self.requests.put(None);break
                items.append(other)
            if self.failed:
                for _,future in items:future.set_exception(self.failed)
                continue
            try:
                resolved={};missing=[]
                for units,_ in items:
                    for unit in units:
                        if unit in resolved:continue
                        value=self.cache.get(unit)
                        resolved[unit]=value
                        if value is None:missing.append(unit)
                if missing:
                    tick=time.perf_counter();values=self.scorer(missing)
                    self.score_seconds+=time.perf_counter()-tick;self.scored+=len(missing);self.calls+=1
                    if len(values)!=len(missing):raise ValueError('Incorrect scorer result length')
                    for unit,value in zip(missing,values):resolved[unit]=value;self.cache.put(unit,value)
                for units,future in items:future.set_result({u:resolved[u] for u in units})
            except BaseException as exc:
                self.failed=exc
                for _,future in items:future.set_exception(exc)

    def close(self):
        self.requests.put(None);self.thread.join()


def _solve_job(documents,service,weight):
    proposed=[proposed_boundaries(d['raw'],'grid8') for d in documents]
    scores=service.score(d['raw'][a:b] for d,nodes in zip(documents,proposed)
                         for a,b in zip(nodes,nodes[1:]))
    graphs=[];fallbacks=0
    for document,nodes in zip(documents,proposed):
        raw=document['raw'];failed=[(a,b) for a,b in zip(nodes,nodes[1:]) if not scores[raw[a:b]].feasible]
        if failed:
            full=boundary_offsets(raw);expanded=set(nodes)
            for a,b in failed:expanded.update(x for x in full if a<x<b)
            nodes=tuple(sorted(expanded));fallbacks+=len(failed)
        graphs.append(DenseCoarseGraph(raw,nodes))
    scores.update(service.score(u for graph in graphs for u in graph.keys if u not in scores))
    rows=[{'index':i,'row':document,'path':graph.solve(scores,weight),'rounds':1}
          for i,(document,graph) in enumerate(zip(documents,graphs))]
    return rows,fallbacks


def solve_coarse_pipeline(documents,scorer,*,score_cache=None,weight=10.,workers=4,
                           group_documents=1024,group_bytes=1024**2):
    if min(workers,group_documents,group_bytes)<1:raise ValueError('Positive pipeline capacities required')
    _dag_path(np.ones((2,1)))
    cache=score_cache if score_cache is not None else FrequentScoreCache(0)
    groups=[];part=[];size=0;start=0
    for index,document in enumerate(documents):
        if part and (len(part)>=group_documents or size+len(document['raw'])>group_bytes):
            groups.append((start,part));start=index;part=[];size=0
        part.append(document);size+=len(document['raw'])
    if part:groups.append((start,part))
    started=time.perf_counter();service=ScoringService(scorer,cache);output=[];fallbacks=0
    try:
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='coarse-cpu') as pool:
            futures=[(start,pool.submit(_solve_job,group,service,weight)) for start,group in groups]
            for start,future in futures:
                rows,expanded=future.result();fallbacks+=expanded
                for row in rows:row['index']+=start
                output.extend(rows)
    finally:service.close()
    return output,{'seconds':time.perf_counter()-started,'scored_candidates':service.scored,
                   'score_wall_seconds':service.score_seconds,'score_calls':service.calls,
                   'fallback_intervals':fallbacks,'cpu_groups':len(groups),'workers':workers}
