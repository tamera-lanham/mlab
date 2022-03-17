import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process

class Worker:
    
    def __init__(self, rank, n_workers, hyperparameters, use_gpu):
        self.rank = rank
        self.ranks = list(range(n_workers))
        self.hyps = hyperparameters
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
        self.data = {}
        self.inputs = {}
        self.outputs = {}
        self.loss = {}
        
    def send(self, data, to_rank):
        group = self.groups[(self.rank, to_rank)]
        
        # First send the shape
        shape = t.tensor(data.shape)
        shape_holder = -t.ones((8,), dtype=t.int32)
        if len(shape) != 0: 
            shape_holder[-len(shape):] = shape
            shape_holder = shape_holder.clone()            
            
        dist.broadcast(shape_holder, self.rank, group)
        
        # Then send the data
        dist.broadcast(data, self.rank, group)
        
    def send_to_all(self, data):
        for rank in self.ranks:
            if rank != self.rank: self.send(data, rank)
        
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
    
                
    def send_multiple(self, tensor_list, to_rank):
        n_tensors = len(tensor_list)
        self.send(t.tensor(n_tensors, dtype=t.int32), to_rank)
        
        for tensor in tensor_list:
            self.send(tensor, to_rank)
    
    def recv_multiple(self, from_rank, dtype=t.float32):
        n_tensors = self.recv(from_rank, t.int32)
        return [self.recv(from_rank, dtype) for _ in range(n_tensors)]

    
def _init_process(func, rank, n_workers, hyperparameters, use_gpu, backend):
    dist.init_process_group(backend, rank=rank, world_size=n_workers)
    worker = Worker(rank, n_workers, hyperparameters, use_gpu)
    func(worker)

def create_workers(func, n_workers, hyperparameters=None, use_gpu=True, backend='gloo'):
                    
    print('Starting %d workers...' % n_workers)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    mp.set_start_method('spawn', force=True)

    processes = []
    for rank in range(n_workers):
        p = mp.Process(
            target=_init_process, 
            args=(func, rank, n_workers, hyperparameters, use_gpu, backend)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()    

    