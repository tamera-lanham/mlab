import torch as t
import dataclasses
import data
import gpt_model
import mlp_model
import training
import workers

@dataclasses.dataclass
class Hyperparameters:
    use_gpt: bool
    n_microbatches: int # For fake data only
    microbatch_size: int
    n_microbatches_per_batch: int
    seq_len: int = 512
    n_epochs: int = 1
    lr: float = 0.01
    
def get_n_batches(w: workers.Worker):
    if w.first:
        n_batches = t.tensor(len(w.data['train'])).int()
        w.send_to_all(n_batches) 
    else:
        n_batches = w.recv(0, t.int32)  
        
    return n_batches

def run(worker):
    # Load the model
    worker.model = mlp_model.load_model(worker.rank).to(worker.device)
    print(worker.rank, worker.model)
    
    # Load the data
    if worker.first: 
        train, test = data.get_data(worker.hyps, fake=True)
        worker.data['train'], worker.data['test'] = train, test
    n_microbatches = get_n_batches(worker)
        
    optimizer = t.optim.Adam(worker.model.parameters())
        
    batch_size = worker.hyps.n_microbatches_per_batch
    
    for epoch in range(worker.hyps.n_epochs):
        for batch_i in range(0, n_microbatches, batch_size):
                
            for microbatch_i in range(batch_i, batch_i + batch_size): 
                training.send_targets(worker, microbatch_i)
                #training.forward_pass(worker, microbatch_i)

            for microbatch_i in range(batch_i, batch_i + batch_size):
                training.forward_pass(worker, microbatch_i)
                loss = training.calc_loss(worker, microbatch_i)
                training.backward_pass(worker, microbatch_i, optimizer, loss)

            break
        break


def main():
    n_gpus =  t.cuda.device_count()

    n_workers = n_gpus if n_gpus else 4
    
    hyps = Hyperparameters(
        use_gpt = False, # If false, use MLP
        seq_len = 128,
        n_microbatches = 1000, # Only applies to fake data
        microbatch_size = 5,
        n_microbatches_per_batch = 5
    )
    
    if hyps.use_gpt:
        raise NotImplementedError()
    else:
        layer_sizes = [hyps.seq_len] + [2 * hyps.seq_len]*(n_workers-1) + [hyps.seq_len]
        model = mlp_model.MLP(layer_sizes, bias=True)
        mlp_model.split_MLP_and_save(model, n_workers)
    
    workers.create_workers(run, n_workers, hyps, use_gpu = bool(n_gpus))


if __name__=="__main__":
    main()
    
    
# conda activate ~/Developer/mlab/env && cd ~/Developer/mlab/days/w3d1/refactor && python run.py

# kill -9 $(lsof -t -i :29500) && python run.py