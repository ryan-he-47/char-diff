"""Small process-scoped NVIDIA sampling log, plus PyTorch peak accounting."""
import collections
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import torch


class GPUMonitor:
    def __init__(self,path,interval=5):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.interval=interval;self.samples=collections.deque(maxlen=120);self.stop=threading.Event()
        self.thread=None;self.error=None;self.uuid=None

    def start(self):
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                       '--format=csv,noheader,nounits'],text=True,timeout=10)
        for row in csv.reader(output.splitlines()):
            if len(row)==2 and row[0].strip()==str(os.getpid()):self.uuid=row[1].strip();break
        if self.uuid is None:raise RuntimeError('Current CUDA process not found in nvidia-smi')
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
        return self

    def _run(self):
        with self.path.open('a',encoding='utf8') as handle:
            while not self.stop.is_set():
                try:
                    output=subprocess.check_output(['nvidia-smi','-i',self.uuid,
                        '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory',
                        '--format=csv,noheader,nounits'],text=True,timeout=10)
                    values=[float(v.strip()) for v in next(csv.reader(output.splitlines()))]
                    sample=dict(zip(('memory_used_mib','memory_total_mib','gpu_percent','memory_busy_percent'),values))
                    sample.update(timestamp=time.time(),pid=os.getpid(),gpu_uuid=self.uuid)
                    self.samples.append(sample);handle.write(json.dumps(sample)+'\n');handle.flush()
                except Exception as exc:self.error=str(exc)
                self.stop.wait(self.interval)

    def summary(self):
        total=torch.cuda.get_device_properties(0).total_memory
        out={'device':torch.cuda.get_device_name(),'total_bytes':total,
             'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'peak_allocated_percent':100*torch.cuda.max_memory_allocated()/total,
             'peak_reserved_percent':100*torch.cuda.max_memory_reserved()/total,
             'sampling_error':self.error}
        rows=list(self.samples)
        if rows:
            out['recent_samples']=len(rows)
            out['nvidia_memory_peak_mib']=max(r['memory_used_mib'] for r in rows)
            out['nvidia_memory_peak_percent']=max(100*r['memory_used_mib']/r['memory_total_mib'] for r in rows)
            out['gpu_utilization_mean']=statistics.mean(r['gpu_percent'] for r in rows)
            out['gpu_utilization_median']=statistics.median(r['gpu_percent'] for r in rows)
            out['gpu_utilization_max']=max(r['gpu_percent'] for r in rows)
        return out

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=12)
