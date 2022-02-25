import math
import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import transformers
import json
from io import StringIO
import time
import transformers

import sys
sys.path.append('/home/ubuntu/mlab/days')
sys.path.append('/home/paperspace/mlab/days')
import utils

DEVICES=[0,1]



def import_lesswrong_corpus(filename="small_lw_corpus.json"):
    with open(filename) as file:
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

def to_batches(list_of_tensors, batch_size):
    n_batches = math.ceil(len(list_of_tensors) / batch_size)
    batches = [list_of_tensors[i:i+batch_size] for i in range(n_batches)]
    return batches

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
            self.corpus = import_lesswrong_corpus()
            list_of_tensors = corpus_to_tensor_list(self.corpus, self.max_tokens)
            self.batches = to_batches(list_of_tensors, self.batch_size)
            self.n_batches = len(self.batches)
            print('I am rank 0, and I know that there are %d batches' % self.n_batches)
            
        else: 
            self.n_batches = 0
            
        # By broadcasting the counter, Now you know at least how long it is, 
        # so you can guess what the batch's shape is like       
        n_batches_holder = t.Tensor([self.n_batches]).to(self.gpu)
        dist.broadcast(n_batches_holder, 0)
        self.n_batches = int(n_batches_holder)
        
        print('After broadcast: I am rank %d, and I know that there are %d batches' % (self.rank, self.n_batches))
        
        
    def __iter__(self): # -> t.tensor generator
        def generator():
            # Hard coding iterations here
            for i in range(self.n_batches):
                
                if self.rank == 0:
                    batch_tensor_list = self.batches[i]
                    batch = t.stack(batch_tensor_list).to(self.gpu) # will have shape (self.batch_size, self.max_tokens)
                
                else:
                    batch = t.zeros((self.batch_size, self.max_tokens), dtype=t.int64).to(self.gpu)

                dist.broadcast(batch, 0, async_op=False) # wait until you get data if you're rank 1
                                   
                # Get the mini-batch from the batch     
                minibatch = batch[self.rank*self.mini_batch_size:(self.rank + 1)*self.mini_batch_size]
                
                yield minibatch
                
        return generator()

# cd mlab/days/w2d5/ && python run.py
def run(rank, size):
    # tensor = t.zeros(3,3)
    group = dist.new_group(list(range(size)))
    mini_batch_size = 16
    dataloader = DistributedDataLoader(rank, size, group, mini_batch_size, random_seed=42)
    
    gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to("cuda:"+str(rank)) 
    optimizer = t.optim.Adam(gpt2.parameters())
    loss_func = t.nn.CrossEntropyLoss()
    
    loss_vals, accs = [], []
    running_loss = 0
    
    for minibatch in dataloader: # first 80% are train, rest are val
        train = minibatch[:math.floor(mini_batch_size*0.8),:]
        val = minibatch[math.floor(mini_batch_size*0.8):,:]
        
        optimizer.zero_grad()
        _, outputs = gpt2(train)
        
        loss = loss_func(outputs, train)
        running_loss += loss.detach()
        mem_usage = mem() # moment of highest GPU usage
        loss.backward()
        
        # Shares the gradients and average it out
        for param in gpt2.parameters():
            temp_grad = param.grad
            dist.all_reduce(temp_grad, dist.ReduceOp.SUM)
            temp_grad /= size
        
        optimizer.step()
        
        
        if i % training_config['val_batches'] == 0:
            with t.no_grad():

                test_targets, test_inputs = zip(*train_batch)
                test_targets = t.Tensor(test_targets).to(device)
                test_inputs = t.stack(test_inputs).to(device)

                _, test_outputs = bert(test_inputs)
                accs.append(float(acc(test_outputs, test_targets).cpu().detach()))
                loss_vals.append(running_loss / training_config['val_batches'])
                running_loss = 0.0 

                data.set_postfix({'acc': sum(accs[-5:]) / min(len(accs), 5), 'loss': float(loss_vals[-1]), 'gpu usage': f'{mem_usage:.1f} GiB'})
                
    # for param in model.parameters():
        # param.grad          
    # randt = t.randn(3,3)
    # sum_ = randt.clone()
    # print(rank, randt)
    # print(rank, randt.sum())
    # dist.all_reduce(sum_, dist.ReduceOp.SUM, group)
    # print('Rank ', rank, ' has data ', sum_)
    # print('The sum of this matrix is ', sum_.sum())

def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    print('I am initiating the process at:', rank, ' of ', size)
    fn(rank, size)
    
def main():
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
      
    
if __name__ == "__main__":
    main()