"""The same bounded two-queue score cache, with batched compiled bookkeeping."""
import json,struct
from pathlib import Path
import numpy as np
from numba import njit,types
from numba.typed import Dict


@njit(cache=True,nogil=True)
def _key(row):return (row[0],row[1],row[2],row[3],row[4])


@njit(cache=True,nogil=True)
def _detach(slot,previous,next_,tier,heads,tails,counts):
    queue=tier[slot];before=previous[slot];after=next_[slot]
    if before>=0:next_[before]=after
    else:heads[queue]=after
    if after>=0:previous[after]=before
    else:tails[queue]=before
    counts[queue]-=1;previous[slot]=-1;next_[slot]=-1


@njit(cache=True,nogil=True)
def _append(slot,queue,previous,next_,tier,heads,tails,counts):
    tail=tails[queue];previous[slot]=tail;next_[slot]=-1;tier[slot]=queue
    if tail>=0:next_[tail]=slot
    else:heads[queue]=slot
    tails[queue]=slot;counts[queue]+=1


@njit(cache=True,nogil=True)
def _evict(queue,index,keys,previous,next_,tier,heads,tails,counts,free,state):
    slot=heads[queue]
    _detach(slot,previous,next_,tier,heads,tails,counts)
    del index[_key(keys[slot])]
    tier[slot]=-1;free[state[0]]=slot;state[0]+=1;state[3]+=1


@njit(cache=True,nogil=True)
def _query(queries,index,keys,values,previous,next_,tier,heads,tails,counts,free,state,length_requests,length_hits,hot_limit):
    output=np.empty((len(queries),2),np.float64);missing=np.empty(len(queries),np.int64);n=0
    for i in range(len(queries)):
        key=_key(queries[i]);length=int(key[0]&np.uint64(255));length_requests[length]+=1
        if key not in index:state[2]+=1;missing[n]=i;n+=1;continue
        slot=index[key]
        output[i]=values[slot]
        if tier[slot]==1:
            _detach(slot,previous,next_,tier,heads,tails,counts);_append(slot,1,previous,next_,tier,heads,tails,counts)
        else:
            _detach(slot,previous,next_,tier,heads,tails,counts)
            _append(slot,1 if hot_limit else 0,previous,next_,tier,heads,tails,counts)
            if hot_limit and counts[1]>hot_limit:_evict(1,index,keys,previous,next_,tier,heads,tails,counts,free,state)
        state[1]+=1;length_hits[length]+=1
    return output,missing[:n]


@njit(cache=True,nogil=True)
def _update(queries,fresh,index,keys,values,previous,next_,tier,heads,tails,counts,free,state,capacity,probation_limit):
    if not capacity:return
    for i in range(len(queries)):
        key=_key(queries[i])
        if key in index:
            slot=index[key]
            queue=tier[slot];_detach(slot,previous,next_,tier,heads,tails,counts)
        else:
            if not state[0]:_evict(0,index,keys,previous,next_,tier,heads,tails,counts,free,state)
            state[0]-=1;slot=free[state[0]];keys[slot]=queries[i];index[key]=slot;queue=0
        values[slot]=fresh[i];_append(slot,queue,previous,next_,tier,heads,tails,counts)
        limit=max(probation_limit,capacity-counts[1])
        while counts[0]>limit:_evict(0,index,keys,previous,next_,tier,heads,tails,counts,free,state)


@njit(cache=True,nogil=True)
def _restore(queries,fresh,tiers,index,keys,values,previous,next_,tier,heads,tails,counts,free,state):
    for i in range(len(queries)):
        state[0]-=1;slot=free[state[0]];keys[slot]=queries[i];values[slot]=fresh[i];index[_key(queries[i])]=slot
        _append(slot,tiers[i],previous,next_,tier,heads,tails,counts)


@njit(cache=True,nogil=True)
def _ordered(heads,next_):
    rows=[]
    for queue in range(2):
        slot=heads[queue]
        while slot>=0:rows.append((queue,slot));slot=next_[slot]
    return rows


def numeric_keys(packed):
    if packed.dtype!=np.uint8 or packed.ndim!=2 or packed.shape[1]!=33:raise ValueError('Expected uint8 (N,33) keys')
    if ((packed[:,0]<1)|(packed[:,0]>32)).any():raise ValueError('Invalid cache key length')
    # Include length and every byte, with zeroed unused suffix. Hash collisions
    # are resolved by equality of all five uint64 words, preserving exact keys.
    padded=np.zeros((len(packed),40),np.uint8);padded[:,:33]=packed
    return padded.view(np.uint64).reshape(-1,5)


class CompiledScoreCache:
    def __init__(self,capacity=500000,scope=None):
        self.capacity=int(capacity)
        if self.capacity<0:raise ValueError('Negative cache capacity')
        self.scope=scope or {};self.probation_limit=max(1,self.capacity//4)
        self.hot_limit=max(0,self.capacity-self.probation_limit)
        n=max(1,self.capacity)
        self.index=Dict.empty(types.UniTuple(types.uint64,5),types.int64)
        self.keys=np.zeros((n,5),np.uint64);self.values=np.empty((n,2),np.float64)
        self.previous=np.full(n,-1,np.int64);self.next=np.full(n,-1,np.int64);self.tier=np.full(n,-1,np.int8)
        self.heads=np.full(2,-1,np.int64);self.tails=np.full(2,-1,np.int64);self.counts=np.zeros(2,np.int64)
        self.free=np.arange(n,dtype=np.int64);self.state=np.asarray([self.capacity,0,0,0],np.int64)
        self.length_requests=np.zeros(33,np.int64);self.length_hits=np.zeros(33,np.int64)
        # Compile outside shard timing, without inserting entries or changing counters.
        empty=np.empty((0,33),np.uint8);self.query(empty);self.update(empty,np.empty((0,2)))
    def _args(self):
        return (self.index,self.keys,self.values,self.previous,self.next,self.tier,self.heads,self.tails,self.counts,self.free,self.state)
    def query(self,packed):
        return _query(numeric_keys(packed),*self._args(),self.length_requests,self.length_hits,self.hot_limit)
    def update(self,packed,values):
        _update(numeric_keys(packed),values,*self._args(),self.capacity,self.probation_limit)
    def statistics(self):
        return {'capacity':self.capacity,'entries':int(self.counts.sum()),'hot_entries':int(self.counts[1]),
                'hits':int(self.state[1]),'misses':int(self.state[2]),'evictions':int(self.state[3]),
                'length_hits':self.length_hits.tolist(),'length_requests':self.length_requests.tolist()}
    def save(self,path):
        header=json.dumps({'scope':self.scope,'capacity':self.capacity,'format':1},sort_keys=True).encode()
        blob=bytearray(struct.pack('<I',len(header))+header)
        rows=self.keys.view(np.uint8).reshape(-1,40)
        for queue,slot in _ordered(self.heads,self.next):
            length=int(rows[slot,0]);feasible,quality=self.values[slot]
            blob.extend(struct.pack('<BBBd',queue,length,int(feasible),quality));blob.extend(rows[slot,1:1+length].tobytes())
        path=Path(path);temporary=path.with_name(path.name+'.pending');path.parent.mkdir(parents=True,exist_ok=True)
        temporary.write_bytes(blob);temporary.replace(path)
    def load(self,path):
        if self.counts.sum():raise ValueError('Load cache into an empty instance')
        blob=Path(path).read_bytes();size=struct.unpack_from('<I',blob)[0];meta=json.loads(blob[4:4+size]);pos=4+size
        if meta['scope']!=self.scope or meta['format']!=1:raise ValueError('Score cache scope mismatch')
        queues=[[],[]]
        while pos<len(blob):
            queue,length,feasible,q=struct.unpack_from('<BBBd',blob,pos);pos+=11;unit=blob[pos:pos+length];pos+=length
            if len(unit)!=length or not 1<=length<=32 or queue not in (0,1):raise ValueError('Invalid cache entry')
            queues[queue].append((unit,feasible,q))
        hot=queues[1][-self.hot_limit:] if self.hot_limit else []
        space=max(0,self.capacity-len(hot));cold=queues[0][-space:] if space else []
        entries=cold+hot
        if entries:
            packed=np.frombuffer(b''.join(bytes((len(u),))+u.ljust(32,b'\0') for u,_,_ in entries),np.uint8).reshape(-1,33)
            values=np.asarray([(ok,q) for _,ok,q in entries],np.float64)
            tiers=np.asarray([0]*len(cold)+[1]*len(hot),np.int8)
            _restore(numeric_keys(packed),values,tiers,*self._args())
