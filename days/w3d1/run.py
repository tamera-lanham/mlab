import dataclasses
import einops
import math
import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import transformers
import json 
import time
import transformers
from tqdm import tqdm

from utils import *

def load_gpt_block(rank):
    return t.load("gpt-j-" + str(rank) + ".pt")

def label_to_tensor(label): 
    if label == 0:
        return t.Tensor([0, 1])
    else: 
        return t.Tensor([1, 0])

    
def get_batches(rank, gpu, fake=True, batch_size=32, max_seq_len=512, n_fake_batches=10):
    
    if rank == 0:
        
        if fake: train_batches, test_batches = fake_imdb_data(batch_size, max_seq_len, n_fake_batches)
        else: train_batches, test_batches = imdb_data(batch_size, max_seq_len)
        
        n_batches = len(train_batches)
        seq_len = int(train_batches[0][0][1].shape[0])
    else:
        train_batches, test_batches = None, None
        n_batches, seq_len = -1, -1
        
    holder = t.Tensor([n_batches, seq_len]).to(gpu)
    dist.broadcast(holder, 0)
    
    n_batches, seq_len = int(holder[0]), int(holder[1])
    
    return n_batches, seq_len, (train_batches, test_batches)


@dataclasses.dataclass
class ProcessInfo: # Class just to hold info for one process
    rank: int
    size: int
    model: t.nn.Module
    gpu: str
    next_group: t._C._distributed_c10d.ProcessGroup
    prev_group: t._C._distributed_c10d.ProcessGroup
    
    def as_tuple(self) -> tuple:
        return tuple(getattr(self, field.name) for field in dataclasses.fields(self))
    

def prepare_process(rank, size, all_groups): # Loads the model and gets the groups for a process
    
    gpu = 'cuda:%d' % rank
    model = load_gpt_block(rank).to(gpu)
    
    next_group = all_groups[rank] if (rank != size-1) else None
    prev_group = all_groups[rank-1] if (rank != 0) else None

    return ProcessInfo(rank, size, model, gpu, next_group, prev_group)


def forward_pass_microbatch(p_info, microbatch=None, middle_shape=(1, 512, 4096)):
    
    (rank, size, model, gpu, next_group, prev_group) = p_info.as_tuple()
    
    # Receive data
    if not prev_group: # When this is the first model in the chain (rank 0)
        input_data = microbatch        
    else:
        input_data = t.zeros(middle_shape).to(gpu)
        dist.broadcast(input_data, rank-1, prev_group)
        
    # Do the forward pass
    output_data = model(input_data)
    
    # Send data
    if next_group: 
        dist.broadcast(output_data, rank, next_group)
        return None
    else: # When this is the last model in the chain
        return output_data
    
def forward_pass_batch(p_info, data_batch, n_microbatches, microbatch_size, seq_len): # bata_batch has data when rank==1, otherwise it's None
    
    middle_shape = (microbatch_size, seq_len, 4096)
    
    microbatches = batch_to_microbatches(data_batch, n_microbatches, microbatch_size) if (data_batch is not None) else None
    
    final_outputs = []
    for i in range(n_microbatches):
        
        microbatch = microbatches[i].to(p_info.gpu) if microbatches else None
        
        output_data = forward_pass_microbatch(p_info, microbatch, middle_shape)
        print(f'rank {p_info.rank}, microbatch {i}, gpu usage {mem(p_info.rank):.2f} GiB')
        
        if output_data is not None:
            final_outputs.append(output_data.detach())
             # This .detach() prevents a GPU memory overflow - but we might actually need to keep the gradients around for the backward pass 😬 
            
    if final_outputs:
        return t.cat(final_outputs, 0)
    

# cd /home/ubuntu/mlab/days/w3d1 && python run.py
def run(rank, size, all_groups): 
    
    microbatch_size = 10
    n_microbatches = 5
    batch_size = microbatch_size * n_microbatches
    
    process_info = prepare_process(rank, size, all_groups)
    print('Prepared process %d' % rank)
    
    n_batches, seq_len, (train, test) = get_batches(rank, process_info.gpu, fake=False, batch_size=batch_size)
    
    for i in range(n_batches):
        
        
        train_labels, train_inputs = zip(*train[i]) if rank == 0 else (None, None)
        test_labels, test_inputs  = zip(*test[i]) if rank == 0 else (None, None)
                        
        forward_pass_outputs = forward_pass_batch(process_info, train_inputs, n_microbatches, microbatch_size, seq_len)
        
        if forward_pass_outputs is not None: print(forward_pass_outputs.shape)
        
        break

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend, rank=rank, world_size=size)
    all_groups = [dist.new_group([x, x+1]) for x in range(size-1)]
    
    fn(rank, size, all_groups)
    
def main():
    size = 4
    processes = []
    mp.set_start_method('spawn', force=True)
    
    print('Initializing %d processes...' % size)
    
    for rank in range(size):
        p = mp.Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()    
        
if __name__ == "__main__":
    main()