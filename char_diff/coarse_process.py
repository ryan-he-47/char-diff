"""Persistent CPU processes, compact score messages and one GPU/cache owner.

Each worker owns its document graphs. Only uint8 candidate matrices and float64
score matrices cross the process boundary; model weights remain in the parent.
"""
from __future__ import annotations
from concurrent.futures import Future
from multiprocessing.connection import wait
import multiprocessing as mp
from multiprocessing import util as _mp_util
import atexit
import hashlib
import math
import queue
import signal
import threading
import time
from typing import NamedTuple

import numpy as np

from .ae_path_runtime import _dag_path,FrequentScoreCache
from .ae_paths import PathResult,boundary_offsets,NoFeasiblePath
from .coarse_pipeline import DenseCoarseGraph
from .path_proposals import proposed_boundaries


class CachedScore(NamedTuple):
    feasible: bool
    quality: float


def pack_candidates(units):
    """One length byte followed by 32 padded bytes; embedded NUL is preserved."""
    if any(not 1<=len(u)<=32 for u in units):raise ValueError('Invalid candidate length')
    return np.frombuffer(b''.join(bytes((len(u),))+u.ljust(32,b'\0') for u in units),np.uint8).reshape(-1,33)


def unpack_candidates(packed):
    blob=packed.tobytes()
    return [blob[i+1:i+1+blob[i]] for i in range(0,len(blob),33)]


def _worker_solve(documents,connection):
    started=time.perf_counter();proposed=[proposed_boundaries(d['raw'],'grid8') for d in documents]
    initial=list(dict.fromkeys(d['raw'][a:b] for d,nodes in zip(documents,proposed) for a,b in zip(nodes,nodes[1:])))
    def score(units):
        if not units:return np.empty((0,2),np.float64)
        connection.send(('score',pack_candidates(units)))
        kind,values=connection.recv()
        if kind!='scores':raise RuntimeError(f'Unexpected worker reply: {kind}')
        if values.shape!=(len(units),2):raise ValueError('Incorrect score matrix shape')
        return values
    values0=score(initial);known={u:i for i,u in enumerate(initial)};graphs=[];fallbacks=0
    for document,nodes in zip(documents,proposed):
        raw=document['raw'];failed=[(a,b) for a,b in zip(nodes,nodes[1:]) if not values0[known[raw[a:b]],0]]
        if failed:
            full=boundary_offsets(raw);expanded=set(nodes)
            for a,b in failed:expanded.update(x for x in full if a<x<b)
            nodes=tuple(sorted(expanded));fallbacks+=len(failed)
        graphs.append(DenseCoarseGraph(raw,nodes))
    lookup={};units=[];indices=[]
    for graph in graphs:
        ids=np.empty(len(graph.keys),np.int64)
        for at,unit in enumerate(graph.keys):
            index=lookup.get(unit)
            if index is None:index=len(units);lookup[unit]=index;units.append(unit)
            ids[at]=index
        indices.append(ids)
    values=np.empty((len(units),2),np.float64);unknown=[];unknown_ids=[]
    for i,u in enumerate(units):
        old=known.get(u)
        if old is None:unknown.append(u);unknown_ids.append(i)
        else:values[i]=values0[old]
    values[unknown_ids]=score(unknown)
    rows=[]
    for document,graph,ids in zip(documents,graphs,indices):
        selected=values[ids];edge_costs=np.where(selected[:,0]!=0,1.+10.*selected[:,1],np.inf)
        costs=np.full((len(graph.nodes),32),np.inf);costs[graph.ends,graph.slots]=edge_costs
        nodes,cost=_dag_path(costs)
        if not len(nodes):
            reachable = np.zeros(len(graph.nodes), dtype=bool)
            reachable[0] = True
            for end, slot, edge_cost in zip(graph.ends, graph.slots, edge_costs):
                if reachable[end - slot - 1] and np.isfinite(edge_cost):
                    reachable[end] = True
            last_byte = graph.nodes[np.flatnonzero(reachable)[-1]]
            digest = hashlib.sha256(graph.raw).hexdigest()
            raise NoFeasiblePath(
                f"grid8 path stopped: file={document.get('source_file')} "
                f"row={document.get('source_row')} language={document.get('language')} "
                f"sha256={digest} total_bytes={len(graph.raw)} furthest_reachable_byte={last_byte}"
            )
        boundaries=tuple(graph.nodes[int(i)] for i in nodes)
        quality=math.fsum(values[lookup[graph.raw[a:b]],1] for a,b in zip(boundaries,boundaries[1:]))
        rows.append((boundaries,float(cost),quality))
    return rows,{'fallback_intervals':fallbacks,'worker_wall_seconds':time.perf_counter()-started}


def _worker_main(connection):
    # Slurm signals the whole step. The parent drains the current shard and
    # then closes workers; children keep serving its outstanding requests.
    for name in ('SIGUSR1','SIGTERM'):
        if hasattr(signal,name):signal.signal(getattr(signal,name),signal.SIG_IGN)
    try:
        _dag_path(np.ones((2,1)))
        connection.send(('ready',None))
        while True:
            kind,payload=connection.recv()
            if kind=='stop':return
            if kind!='solve':raise RuntimeError(f'Unexpected worker command: {kind}')
            connection.send(('result',_worker_solve(payload,connection)))
    except EOFError:pass
    except BaseException:
        import traceback
        try:connection.send(('error',traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:connection.close()


class ArrayScoreService:
    def __init__(self,scorer,cache):
        self.scorer=scorer;self.cache=cache;self.requests=queue.Queue();self.failed=None
        self.scored=0;self.calls=0;self.score_seconds=0.;self.cache_seconds=0.
        self.thread=threading.Thread(target=self._run,name='array-score',daemon=True);self.thread.start()
    def submit(self,packed):
        future=Future();self.requests.put((packed,future));return future
    def _run(self):
        while True:
            item=self.requests.get()
            if item is None:return
            packed,future=item
            if self.failed:future.set_exception(self.failed);continue
            try:
                tick=time.perf_counter();units=None
                if hasattr(self.cache,'query'):values,missing=self.cache.query(packed)
                else:
                    units=unpack_candidates(packed)
                    values=np.empty((len(units),2),np.float64);missing=[];hit_ids=[];hits=[]
                    for i,unit in enumerate(units):
                        value=self.cache.get(unit)
                        if value is None:missing.append(i)
                        else:hit_ids.append(i);hits.append((value.feasible,value.quality))
                    if hit_ids:values[hit_ids]=np.asarray(hits)
                self.cache_seconds+=time.perf_counter()-tick
                if len(missing):
                    tick=time.perf_counter()
                    if hasattr(self.scorer,'score_packed'):fresh=self.scorer.score_packed(packed[missing])
                    else:
                        if units is None:units=unpack_candidates(packed)
                        fresh=np.asarray([(v.feasible,v.quality) for v in self.scorer([units[i] for i in missing])],np.float64)
                    self.score_seconds+=time.perf_counter()-tick;self.calls+=1;self.scored+=len(missing)
                    if fresh.shape!=(len(missing),2) or not np.isfinite(fresh).all() or (fresh[:,1]<0).any():
                        raise FloatingPointError('Invalid packed AE scores')
                    tick=time.perf_counter();values[missing]=fresh
                    if hasattr(self.cache,'update'):self.cache.update(packed[missing],fresh)
                    else:
                        for i,(ok,q) in zip(missing,fresh.tolist()):self.cache.put(units[i],CachedScore(bool(ok),q))
                    self.cache_seconds+=time.perf_counter()-tick
                future.set_result(values)
            except BaseException as exc:self.failed=exc;future.set_exception(exc)
    def close(self):self.requests.put(None);self.thread.join()


class ProcessCoarseSolver:
    """Keep spawn workers alive across shards; GPU inference stays in parent."""
    def __init__(self,workers=4):
        if workers<1:raise ValueError('Expected positive worker count')
        self.workers=workers;self.processes=[];self.connections=[];self.closed=False
        atexit.register(self.close)
        context=mp.get_context('spawn')
        try:
            for _ in range(workers):
                parent,child=context.Pipe()
                process=context.Process(target=_worker_main,args=(child,),daemon=True)
                process.start();child.close();self.processes.append(process);self.connections.append(parent)
            for connection in self.connections:
                if not connection.poll(120):raise RuntimeError('CPU worker startup timed out')
                kind,value=connection.recv()
                if kind!='ready':raise RuntimeError(f'CPU worker startup: {kind}: {value}')
        except BaseException:self.close();raise

    def solve(self,documents,scorer,*,score_cache=None,group_documents=1024,group_bytes=1024**2):
        if self.closed:raise RuntimeError('CPU pool is closed')
        if min(group_documents,group_bytes)<1:raise ValueError('Positive group limits required')
        groups=[];part=[];size=0;start=0
        for i,document in enumerate(documents):
            if part and (len(part)>=group_documents or size+len(document['raw'])>group_bytes):
                groups.append((start,part));part=[];size=0;start=i
            part.append(document);size+=len(document['raw'])
        if part:groups.append((start,part))
        cache=score_cache if score_cache is not None else FrequentScoreCache(0)
        service=ArrayScoreService(scorer,cache);started=time.perf_counter();pending={};active={};outputs={};next_group=0
        stats={'fallback_intervals':0,'worker_wall_seconds':0.}
        def assign(connection):
            nonlocal next_group
            if next_group>=len(groups):return
            active[connection]=next_group;connection.send(('solve',groups[next_group][1]));next_group+=1
        try:
            for connection in self.connections:assign(connection)
            while active:
                for connection,future in list(pending.items()):
                    if future.done():connection.send(('scores',future.result()));del pending[connection]
                readable=[c for c in active if c not in pending]
                ready=wait(readable,timeout=.01) if readable else []
                if not ready:
                    if not readable:time.sleep(.001)
                    for process in self.processes:
                        if not process.is_alive():raise RuntimeError(f'CPU worker exited: {process.exitcode}')
                for connection in ready:
                    kind,value=connection.recv()
                    if kind=='score':pending[connection]=service.submit(value)
                    elif kind=='result':
                        index=active.pop(connection);outputs[index]=value[0]
                        for key,number in value[1].items():stats[key]+=number
                        assign(connection)
                    else:raise RuntimeError(f'CPU worker {kind}: {value}')
            rows=[]
            for i,(start,group) in enumerate(groups):
                for offset,(document,(boundaries,cost,quality)) in enumerate(zip(group,outputs[i])):
                    units=tuple(document['raw'][a:b] for a,b in zip(boundaries,boundaries[1:]))
                    rows.append({'index':start+offset,'row':document,'path':PathResult(boundaries,units,cost,quality),'rounds':1})
        except BaseException:
            # Workers may be awaiting scores; terminate only this owned pool.
            self.close();raise
        finally:service.close()
        return rows,{'seconds':time.perf_counter()-started,'scored_candidates':service.scored,
                     'score_wall_seconds':service.score_seconds,'cache_seconds':service.cache_seconds,
                     'score_calls':service.calls,'cpu_groups':len(groups),'workers':self.workers,**stats}

    def close(self):
        if self.closed:return
        self.closed=True
        atexit.unregister(self.close)
        for connection in self.connections:
            try:connection.send(('stop',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():process.kill();process.join(timeout=2)
        for connection in self.connections:connection.close()
