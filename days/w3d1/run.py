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


def get_batches(rank, gpu, fake=True, batch_size=32, max_seq_len=512, n_fake_batches=10):
    
    if rank == 0:
        
        if fake: train_batches, test_batches = fake_imdb_data(batch_size, max_seq_len, n_fake_batches)
        else: train_batches, test_batches = imdb_data(batch_size, max_seq_len)
        
        n_batches = len(train_batches)
        seq_len = int(train_batches[0][1][1].shape[0])
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
    #gpu = 'cpu'
    model = load_gpt_block(rank).to(gpu)
    
    next_group = all_groups[rank] if (rank != size-1) else None
    prev_group = all_groups[rank-1] if (rank != 0) else None

    return ProcessInfo(rank, size, model, gpu, next_group, prev_group)

def pass_to_last(p_info, data, group, tensor_shape = (32,)): # data is either labels or n_real_tokens

    if p_info.rank == 0:
        tensor = t.tensor(data, device = p_info.gpu)
        print(f'* {tensor.shape}, {tensor.dtype}, {tensor_shape}')
        
        dist.broadcast(tensor, 0, group)
        
    elif p_info.rank == p_info.size - 1: # If you are the last process
        tensor = t.zeros(tensor_shape, dtype=t.int64, device=p_info.gpu)
        dist.broadcast(tensor, 0, group)
        
    else: tensor = None
        
    return tensor

def forward_pass_microbatch(p_info, microbatch=None, middle_shape=(1, 512, 4096)):
    
    (rank, size, model, gpu, next_group, prev_group) = p_info.as_tuple()
    
    # Receive data
    if not prev_group: # When this is the first model in the chain (rank 0)
        input_data = microbatch        
    else:
        input_data = t.zeros(middle_shape, device=gpu)
        dist.broadcast(input_data, rank-1, prev_group)
        
    # Do the forward pass
    output_data = model(input_data)

    # Send data
    if next_group: 
        dist.broadcast(output_data, rank, next_group)
        return None
    else: # When this is the last model in the chain
        output = output_data[:,:,0] # Why only keep one of the two outputs? Because our sentiment classification task only involves one class, not two classes.
        return output


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
    
def calculate_loss(labels, outputs, n_real_tokens):
    
    loss_fn = t.nn.CrossEntropyLoss()
    
    
def backwards_pass_microbatch(p_info, microbatch_labels, microbatch_outputs, optimizer, middle_shape=(1, 512, 4096)):
    
    (rank, size, model, gpu, next_group, prev_group) = p_info.as_tuple()
    
    optimizer.zero_grad()

    if rank == size-1:
        loss_fn = t.nn.CrossEntropyLoss()
        loss = loss_fn(microbatch_labels, microbatch_outputs)
        loss.backward()
        earliest_layer_grad = get_earliest_layer_grad()
        dist.broadcast(earliest_layer_grad, rank, prev_group)

    elif rank != 0:
        latest_layer_gradient = t.zeros(middle_shape)
        dist.broadcast(latest_layer_gradient, rank+1, next_group)
        latest_layer.backward(latest_layer_grad)
        earliest_layer_grad = get_earliest_layer_grad()
        dist.broadcast(earliest_layer_grad, rank, prev_group)

    else: #rank 0
        latest_layer_gradient = t.zeros(middle_shape)
        dist.broadcast(latest_layer_gradient, rank+1, next_group)
        latest_layer.backward(latest_layer_grad)

    optimizer.step()


def backwards_pass_batch(p_info, batch_labels, batch_outputs, batch_n_tokens, optimizer, n_microbatches, microbatch_size, seq_len):
    
    middle_shape = (microbatch_size, seq_len, 4096)
    
    if p_info.rank == p_info.size - 1:
        print(batch_labels.shape, batch_outputs.shape, batch_n_tokens.shape) # tensor[20,1] tensor[20,1], tensor[20,1] 
        microbatch_labels = batch_to_microbatches(list(batch_labels), n_microbatches, microbatch_size)
        microbatch_outputs = batch_to_microbatches(list(batch_outputs), n_microbatches, microbatch_size)
        microbatch_n_tokens = batch_to_microbatches(list(batch_n_tokens), n_microbatches, microbatch_size)
        
    else:
        microbatch_labels = None
        microbatch_outputs = None
        microbatch_n_tokens = None
    
    for i in range(n_microbatches):
        backwards_pass_microbatch(p_info, microbatch_labels, microbatch_outputs, optimizer, middle_shape)
        

# cd /home/ubuntu/mlab/days/w3d1 && python run.py
def run(rank, size, all_groups): 
    
    microbatch_size = 2
    n_microbatches = 5
    batch_size = microbatch_size * n_microbatches
    
    process_info = prepare_process(rank, size, all_groups)
    print('Prepared process %d' % rank)
    
    optimizer = t.optim.Adam(process_info.model.parameters()) # hyperparams
    
    n_batches, seq_len, (train, test) = get_batches(rank, process_info.gpu, fake=True, batch_size=batch_size)
    
    forward_pass_outputs = []
    
    for i in range(n_batches):
        
        train_labels, train_inputs, train_n_tokens = zip(*train[i]) if rank == 0 else (None, None, None)
        test_labels, test_inputs, test_n_tokens    = zip(*test[i])  if rank == 0 else (None, None, None)
        
        
        train_labels, test_labels, train_n_tokens, test_n_tokens = [
            pass_to_last(process_info, data, all_groups['first_and_last'], (batch_size,))
            for data in [train_labels, test_labels, train_n_tokens, test_n_tokens]
        ]
       
        forward_pass_outputs = forward_pass_batch(process_info, train_inputs, n_microbatches, microbatch_size, seq_len)

        if rank == size - 1:
            print(f'! {train_labels.shape}, {train_labels.dtype}, {train_n_tokens.shape}, {train_n_tokens.dtype}')
        
        
        # Begins the backwards pass
        backwards_pass_batch(process_info, train_labels, forward_pass_outputs, train_n_tokens, optimizer, n_microbatches, microbatch_size, seq_len)
        
        break # albeit, only for the first batch
        
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend, rank=rank, world_size=size)
    
    all_groups = {x: dist.new_group([x, x+1]) for x in range(size-1)}
    all_groups['first_and_last'] = dist.new_group([0, size - 1])
    
    fn(rank, size, all_groups)
    
def main():
    size = 8
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