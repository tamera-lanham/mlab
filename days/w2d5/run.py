import math
import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import transformers
import json
from io import StringIO

import sys
sys.path.append('/home/ubuntu/mlab/days')
import utils

DEVICES=[0,1]



def import_lesswrong_corpus():
    with open("/home/ubuntu/lw_corpus.json") as file:
        return json.load(file)
    return None
        

def corpus_to_tensor_list(corpus, max_tokens=1024):
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    
    tensor_list = []
    staged_tokens = []
    
    all_tokens = [token for d in corpus for token in tokenizer(d['title'] + d['text'] + '<|endoftext|>').input_ids]
    for batch in range(math.ceil(len(all_tokens) / max_tokens)):
        start, end = batch * max_tokens, (batch + 1) * max_tokens
        tensor_list.append(t.tensor(all_tokens[start:end]))
        
    return tensor_list

class DistributedDataLoader():
    def __init__(
        self, 
        rank : int, 
        world_size : int, 
        mini_batch_size : int, 
        random_seed):
        self.rank = rank
        self.world_size = world_size
        self.mini_batch_size = mini_batch_size
        # 1. Check if process is in charge (0 is in charge)
        if rank == 0:
            # 2. If in charge, then load all the training data
            self.corpus = import_lesswrong_corpus()
            list_of_tensors = corpus_to_tensor_list(self.corpus)
            batch_size = self.world_size * self.mini_batch_size
            self.batches = utils.to_batches(list_of_tensors, batch_size)
            self.counter = len(self.batches)
            print(len(self.batches))
            print(len(self.batches[0]))
        else: 
            self.counter = 0
            self.corpus = None
            self.batches = None
        # By broadcasting the counter, Now you know at least how long it is, 
        # so you can guess what the batch's shape is like
        counter_holder = t.Tensor([self.counter])
        dist.broadcast(counter_holder, 0)
        self.counter = int(counter_holder)
        
    def __iter__(self): # -> t.tensor generator
        if self.rank == 0:
            # 3. dist_all/scatter to all processes
            for batch in self.batches:
                dist.scatter(batch, 0,) #wait where do i get the group?
                yield batch
        else: 
            #4. If i am a second class citizen, I should jsut wait and receive?
            # Wait, how would i know how how long the corpus is? 
            # And How would i hold it if i don't know what the batche's shape is like?
            for batch in range(self.batches):
                batch = t.rand() # how do i guess the size?
                dist.scatter(batch, 0, ) #wait where do i get the group?
                yield batch

def run(rank, size):
    # tensor = t.zeros(3,3)
    group = dist.new_group(DEVICES)
    dataloader = DistributedDataLoader(0, 2, 16, 42)
    # randt = t.randn(3,3)
    # sum_ = randt.clone()
    # print(rank, randt)
    # print(rank, randt.sum())
    # dist.all_reduce(sum_, dist.ReduceOp.SUM, group)
    # print('Rank ', rank, ' has data ', sum_)
    # print('The sum of this matrix is ', sum_.sum())

def init_processes(rank, size, fn, backend='Gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    print('I am initiating the process at:', rank, ' of ', size)
    fn(rank, size)
    
if __name__ == "__main__":
    size = len(DEVICES)
    processes = []
    mp.set_start_method('spawn', force=True)
    for rank in range(size):
        p = mp.Process(target=init_processes, args=(rank, size, run))
        print("I am starting!", rank, size)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()    
    

        
