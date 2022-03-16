import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process

class Worker:
    
    def __init__(self, rank, n_workers, use_gpu):
        self.rank = rank
        self.device = f'cuda:{rank}' if use_gpu else 'cpu'
        
        self.first, self.middle, self.last = False, False, False
        if rank == 0: self.first = True
        elif rank == (n_workers - 1): self.last = True
        else: self.middle = True
        
        groups = {
            (a, b): dist.new_group([a, b])
            for a in range(n_workers) for b in range(n_workers)
            if a < b
        }
        self.groups = {**{(b, a): groups[(a,b)] for a,b in groups.keys()}, **groups}

        self.prev = rank - 1 if (rank != 0) else None
        self.next = rank + 1 if (rank != n_workers-1) else None
        
        self.model = None
        self.tensors = []
        
    def send(self, data, to_rank):
        group = self.groups[(self.rank, to_rank)]
        
        # First send the shape
        shape = t.tensor(data.shape)
        shape_holder = -t.ones((8,), dtype=t.int32)
        shape_holder[-len(shape):] = shape
        dist.broadcast(shape_holder, self.rank, group)
        
        # Then send the data
        dist.broadcast(data, self.rank, group)
        
        
    def recv(self, from_rank, dtype=t.float32):
        group = self.groups[(self.rank, from_rank)]
        
        # First get the shape
        shape_holder = -t.ones((8,), dtype=t.int32)
        dist.broadcast(shape_holder, from_rank, group)
        shape = shape_holder[shape_holder != -1].tolist()
        
        # Then get the data
        data = t.zeros(shape, dtype=dtype, device=self.device)
        dist.broadcast(data, from_rank, group)
        
        return data

    
# This function really should be in Workers, but dist doesn't like that :(
def _init_process(rank, n_workers, use_gpu, backend, func):
    dist.init_process_group(backend, rank=rank, world_size=n_workers)

    worker = Worker(rank, n_workers, use_gpu)

    func(worker)

class Workers:
    
    def __init__(self, n_workers, backend, use_gpu=True):
        self.n_workers = n_workers
        self.backend = backend
        self.use_gpu = use_gpu
        self.processes = []
        
        
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
    
    
def create_workers(n_workers, func, use_gpu=True, backend = 'gloo'):
    workers = Workers(n_workers, backend, use_gpu)
    workers.start(func)
    return workers.workers
    