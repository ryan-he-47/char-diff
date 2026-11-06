"""A bounded, single-owner CPU producer. Checkpoints contain only consumed batches."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np

from .ae_full_sampler import FullCorpusSampler

_worker_sampler = None


def _start(manifest, options, state):
    global _worker_sampler
    _worker_sampler = FullCorpusSampler(manifest, **options)
    _worker_sampler.load_state_dict(state)


def _prepare(count):
    sampler = _worker_sampler
    units = []
    while len(units) < count and not sampler.completed_epochs:
        units.append(sampler.sample().data)
    packed = b"".join(bytes((len(u),)) + u.ljust(32, b"\0") for u in units)
    return np.frombuffer(packed, dtype=np.uint8).reshape(-1, 33), sampler.state_dict()


class PrefetchedEpoch:
    def __init__(self, manifest, options):
        self.manifest = manifest
        self.options = options
        source = FullCorpusSampler(manifest, **options)
        self.protocol = {**source.protocol, "stream": "one_complete_source_pass_v1"}
        self.total_bytes = source.spec["raw_bytes"]
        self.committed = source.state_dict()
        source.close()
        self.executor = None
        self.pending = deque()

    @property
    def stats(self):
        return self.committed["stats"]

    def state_dict(self):
        return self.committed

    def load_state_dict(self, state):
        if self.executor is not None or state["protocol"] != self.committed["protocol"]:
            raise ValueError("Incompatible or active full-data stream")
        self.committed = state

    def next_batch(self, count):
        if self.executor is None:
            self.executor = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                               initializer=_start, initargs=(self.manifest,self.options,self.committed))
            for _ in range(3):
                self.pending.append(self.executor.submit(_prepare,count))
        result = self.pending.popleft().result()
        self.pending.append(self.executor.submit(_prepare,count))
        return result

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)


def epoch_lr(end_byte, total, peak, floor):
    if not 0 <= end_byte <= total or total <= 0:
        raise ValueError("Invalid one-pass progress")
    if end_byte == total:
        return floor
    decay = max(0., (10 * end_byte - 9 * total) / total)
    return peak + (floor - peak) * decay
