import torch as t
import mlp_model
import gpt_model
import time

def forward_pass(worker, microbatch_i, train=True):
    
    if not train: raise NotImplementedError()
    
    # Get inputs
    if worker.first: 
        inputs, _, _ = worker.data['train'][microbatch_i]
        if not worker.hyps.use_gpt: inputs = inputs.float() # MLP needs a float input 
    else: 
        inputs = worker.recv(worker.prev)
        inputs.requires_grad = True
    
    worker.inputs[microbatch_i] = inputs
    
    # Run the forward pass 
    outputs = worker.model(inputs)
    
    worker.outputs[microbatch_i] = outputs

    # Send or return outputs
    if not worker.last: worker.send(outputs, worker.next)
    if worker.last: return outputs
        
def send_targets(worker, microbatch_i):
    # Send the targets and last_token_indices to the last process
    if worker.first:
        _, targets, last_token_indices = worker.data['train'][microbatch_i]
        worker.send(targets, worker.ranks[-1])
        worker.send(last_token_indices, worker.ranks[-1])
    
    if worker.last:
        worker.data[('targets', microbatch_i)] = worker.recv(0, t.int32)
        worker.data[('last_token_indices', microbatch_i)] = worker.recv(0, t.int32)

def calc_loss(worker, microbatch_i):

    if not worker.last: return # Only the last process calculates loss
    
    targets = worker.data[('targets', microbatch_i)]
    last_token_indices = worker.data[('last_token_indices', microbatch_i)]
          
    # Hacky fix for the problem described in reproduce-bug.ipynb
    outputs = worker.model(worker.inputs[microbatch_i])
    # outputs = worker.outputs[microbatch_i]
        
    if worker.hyps.use_gpt: 
        raise NotImplementedError()
    else: 
        return mlp_model.calc_loss(outputs, targets)

    
def backward_pass(worker, microbatch_i, optimizer, loss):
    
    optimizer.zero_grad()
    
    if worker.last:
        loss.backward()
        
    else:
        # Hacky fix for the problem described in reproduce-bug.ipynb
        outputs = worker.model(worker.inputs[microbatch_i])
        #outputs = worker.outputs[microbatch_i]
        
        output_grad = worker.recv(worker.next)
        outputs.backward(output_grad)
        
    if not worker.first:
        input_grad = worker.inputs[microbatch_i].grad
        worker.send(input_grad, worker.prev)
    else:
        time.sleep(1)
        
        
    optimizer.step()
    
    del worker.inputs[microbatch_i], worker.outputs[microbatch_i]
    
    