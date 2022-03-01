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

def fake_imdb(batch_size=32, n_batches=10, max_len=512, vocab_size=50257):
    
    def sample():
        sentiment = int(t.randint(0, 2, (1,)))
        tokens = t.randint(0, vocab_size, (max_len,)).long()
        return sentiment, tokens
    
    train_batches = [[sample() for _ in range(batch_size)] for _ in range(n_batches)]
    test_batches = [[sample() for _ in range(batch_size)] for _ in range(n_batches)]
    
    return train_batches, test_batches

def label_to_tensor(label): 
    if label == 0:
        return t.Tensor([0, 1])
    else: 
        return t.Tensor([1, 0])

class DistributedDataLoader():
    def __init__(
        self, 
        rank : int, 
        world_size : int, 
        group,
        mini_batch_size : int, 
        random_seed):
        
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.mini_batch_size = mini_batch_size

        self.gpu = 'cuda:%d' % self.rank
        
        self.batch_size = self.world_size * self.mini_batch_size
        self.max_tokens=1024
        
        # 1. Check if process is in charge (0 is in charge)
        if rank == 0:
            # 2. If in charge, then load all the training data
            train, test = fake_imdb_data(self.batch_size)
            # self.train_batches = list(to_batches(train, self.batch_size))
            # self.test_batches = list(to_batches(test, self.batch_size))
            self.train_batches = train
            self.test_batche = test
            self.n_train_batches = len(self.train_batches)
            self.n_test_batches = len(self.test_batche)
            
        else: 
            self.n_train_batches = 0
            self.n_test_batches = 0
            
        # By broadcasting the counter, Now you know at least how long it is, 
        # so you can guess what the batch's shape is like       
        # UPDATE 2022.03.01 - Added num for both train batches and test batches
        n_train_batches_holder = t.Tensor([self.n_train_batches]).to(self.gpu)
        n_test_batches_holder = t.Tensor([self.n_test_batches]).to(self.gpu)
        dist.broadcast(n_train_batches_holder, 0)
        dist.broadcast(n_test_batches_holder, 0)
        self.n_train_batches = int(n_train_batches_holder)
        self.n_test_batches = int(n_test_batches_holder)

        
    def __iter__(self): # -> t.tensor generator
        def generator():
            # Hard coding iterations here
            for i in range(self.n_train_batches):
                if self.rank == 0:
                    train_batch = self.train_batches[i].to(self.gpu) 
                        # will have shape (self.batch_size, self.max_tokens)
                    test_batch = self.test_batches[i].to(self.gpu) 
                        # will have shape (self.batch_size, self.max_tokens)
                    
                    
                else:
                    train_batch = t.zeros((self.batch_size, self.max_tokens), dtype=t.int64).to(self.gpu)
                    test_batch = t.zeros((self.batch_size, self.max_tokens), dtype=t.int64).to(self.gpu)

                dist.broadcast(train_batch, 0, async_op=False) # wait until you get data if you're rank >0
                dist.broadcast(test_batch, 0, async_op=False) # wait until you get data if you're rank >0

                # Get the mini-batch from the batch     
                train_minibatch = batch[self.rank*self.mini_batch_size:(self.rank + 1)*self.mini_batch_size]
                test_minibatch = batch[self.rank*self.mini_batch_size:(self.rank + 1)*self.mini_batch_size]

                yield (train_minibatch, test_minibatch)
                
        return generator()


def run(rank, size): 
    group = dist.new_group(list(range(size)))
    mini_batch_size = 16
    dataloader = DistributedDataLoader(rank, size, group, mini_batch_size, random_seed = 42)
    
    # TODO: Write a helper function that hands run func the approrpiate split_model
    partial_gpt = load_gpt_block(rank)
    optimizer = t.optim.SGD(partial_gpt.parameters())
    loss_func = t.nn.BCE()
    
    train_losses, val_losses = [], []
    counter = 0

    for minibatch in dataloader: # first 80% are train, rest are val 
        counter += 1
        # train = minibatch[:math.floor(mini_batch_size*0.8),:] # will have shape (mini_batch_size, max_tokens)
        # val = minibatch[math.floor(mini_batch_size*0.8):,:]
        train, val = minibatch 
        
        # TODO: Understand where the label sits in the training data
        labels, inputs = zip(*train)
        optimizer.zero_grad()
        
        outputs = partial_gpt(inputs)
        print(f'Rank: {rank}, batch: {counter}, memory: {mem(rank)} GiB')
        
        loss = loss_func(outputs, [label_to_tensor(label) for label in labels])
        train_losses += [loss.detach()]
        loss.backward()
        
        optimizer.step()
        
        with t.no_grad():
            val_inputs, val_labels = zip(*val)
            val_outputs = partial_gpt(val_inputs)
            
            val_loss = loss_func(val_outputs, val_labels)
            val_losses.append(val_loss.detach())


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    print('I am initiating the process at:', rank, ' of ', size)
    fn(rank, size)
    
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