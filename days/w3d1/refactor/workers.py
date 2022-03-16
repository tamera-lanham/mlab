import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process

class WorkerInfo:
    
    def __init__(self, rank, n_workers, groups, use_gpu):
        self.rank = rank
        self.device = f'cuda:{rank}' if use_gpu else 'cpu'
        
        self.first, self.middle, self.last = False, False, False
        if rank == 0: self.first = True
        elif rank == (n_workers - 1): self.last = True
        else: self.middle = True
        
        self.groups = groups
        self.prev = self.groups[(rank-1, rank)] if (rank != 0) else None
        self.next = self.groups[(rank, rank+1)] if (rank != n_workers-1) else None
        
        self.model = None
        self.tensors = []

def _init_process(rank, n_workers, use_gpu, backend, func):
    dist.init_process_group(backend, rank=rank, world_size=n_workers)

    groups = {(x, x+1): dist.new_group([x, x+1]) for x in range(n_workers-1)}
    groups[(0, n_workers - 1)] = dist.new_group([0, n_workers - 1])

    worker_info = WorkerInfo(rank, n_workers, groups, use_gpu)

    func(worker_info)

class Workers:
    
    def __init__(self, n_workers, backend, use_gpu=True):
        self.n_workers = n_workers
        self.backend = backend
        self.use_gpu = use_gpu
        
        self.processes = []
        self.groups = {}
        
        self.workers = [None] * self.n_workers
        
    def start(self, func):
                    
        print('Starting %d workers...' % self.n_workers)
        
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        mp.set_start_method('spawn', force=True)
        
        for rank in range(self.n_workers):
            p = mp.Process(target=_init_process, args=(rank, self.n_workers, self.use_gpu, self.backend, func))
            p.start()
            self.processes.append(p)
        for p in self.processes:
            p.join()    
    
    
def create_workers(n_workers, func, backend = 'gloo'):
    workers = Workers(n_workers, backend)
    workers.start(func)
    return workers.workers
    