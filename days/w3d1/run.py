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


def get_batches(rank, gpu):
    if rank == 0:
        train_batches, test_batches = fake_imdb_data()
        n_batches = len(train_batches)
        seq_len = int(train_batches[0][0][1].shape[0])
    else:
        train_batches, test_batches = None, None
        n_batches, seq_len = -1, -1
        
    holder = t.Tensor([n_batches, seq_len]).to(gpu)
    dist.broadcast(holder, 0)
    
    n_batches, seq_len = int(holder[0]), int(holder[1])
    
    return n_batches, seq_len, (train_batches, test_batches)


def forward_pass(rank, size, model, gpu, send_group, recv_group, initial_inputs=None, middle_shape=(1, 512, 4096)):
    
    print(f'Rank: {rank}, memory: {mem(rank)} GiB')
    # Receive data
    if not recv_group: # When this is the first model in the chain (rank 0)
        input_data = initial_inputs        
        print('Rank 0 input shape:', input_data.shape)
    else:
        input_data = t.zeros(middle_shape).to(gpu)
        dist.broadcast(input_data, rank-1, recv_group)
        print('GPU %s received input shape:' %gpu, input_data.shape)
        
    print(f'Rank: {rank}, memory: {mem(rank)} GiB')
    # Do the forward pass
    output_data = model(input_data)
    
    print(f'Rank: {rank}, memory: {mem(rank)} GiB')
    # Send data
    if send_group: 
        dist.broadcast(output_data, rank, send_group)
        print('GPU %s sending input shape:' %gpu, output_data.shape)
        return None
        
    else: # When this is the last model in the chain
        print('Last process returning output data:', output_data.shape)
        return output_data
    
    
def batch_to_microbatches(batch,): # batch is list of [t.tensor()] 
    
    
# cd /home/ubuntu/mlab/days/w3d1 && python run.py
def run(rank, size, all_groups): 
    
    microbatch_size = 16
    batch_size = microbatch_size * size
    
    gpu = 'cuda:%d' % rank

    n_batches, seq_len, (train, test) = get_batches(rank, gpu)
    
    print('rank %d got n_batches %d and seq_len %d' % (rank, n_batches, seq_len))
    
    send_group = all_groups[rank] if (rank != size-1) else None
    recv_group = all_groups[rank-1] if (rank != 0) else None
    
    print('rank %d got groups' % rank)
    
    model = load_gpt_block(rank).to(gpu)
    print('rank %d loaded model' % rank)
    
    for i in range(n_batches):
        
        if rank == 0:
            train_labels, train_inputs = zip(*train[i])
            train_labels = t.tensor(train_labels).to(gpu)
            train_inputs = t.stack(train_inputs).to(gpu)
            
            test_labels, test_inputs = zip(*test[i])
            test_labels = t.tensor(test_labels).to(gpu)
            test_inputs = t.stack(test_inputs).to(gpu)
        
        initial_inputs = train_inputs if rank==0 else None
        
        print(f'Rank: {rank}, memory: {mem(rank)} GiB')
        
        output_data = forward_pass(rank, size, model, gpu, send_group, recv_group, initial_inputs, middle_shape = (1, seq_len, 4096))
        
        if rank==(size - 1):
            print(output_data.shape)
            
        break
        

#         train, val = minibatch 
        
#         # TODO: Understand where the label sits in the training data
#         labels, inputs = zip(*train)
#         optimizer.zero_grad()
        
#         outputs = partial_gpt(inputs)
#         print(f'Rank: {rank}, batch: {counter}, memory: {mem(rank)} GiB')
        
#         loss = loss_func(outputs, [label_to_tensor(label) for label in labels])
#         train_losses += [loss.detach()]
#         loss.backward()
        
#         optimizer.step()
        
#         with t.no_grad():
#             val_inputs, val_labels = zip(*val)
#             val_outputs = partial_gpt(val_inputs)
            
#             val_loss = loss_func(val_outputs, val_labels)
#             val_losses.append(val_loss.detach())


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    all_groups = [dist.new_group([x, x+1]) for x in range(size-1)]
    
    print('I am initiating the process at:', rank, ' of ', size)
    fn(rank, size, all_groups)
    
def main():
    size = 4
    processes = []
    mp.set_start_method('spawn', force=True)
    
    for rank in range(size):
        print(rank)
        p = mp.Process(target=init_processes, args=(rank, size, run))
        print("I am starting!", rank, size)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()    
        
if __name__ == "__main__":
    main()