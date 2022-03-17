import torch as t
import mlp_model
import gpt_model

def forward_pass(worker, microbatch_i, train=True):
    
    if not train: raise NotImplementedError()
    
    # Get inputs
    if worker.first: 
        inputs, _, _ = worker.data['train'][microbatch_i]
        if not worker.hyps.use_gpt: inputs = inputs.float() # MLP needs a float input 
    else: 
        inputs = worker.recv(worker.prev)
        inputs.requires_grad = True
        print('input', worker.rank, inputs._version)
    
    worker.inputs[microbatch_i] = inputs
    
    # Run the forward pass 
    print(worker.rank, 'Forward pass, microbatch %d' % microbatch_i)
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

def calc_loss(worker, microbatch_i):

    if not worker.last: return # All the non-last processes can leave now

    targets = worker.recv(0, t.int32)
    last_token_indices = worker.recv(0, t.int32)
    outputs = worker.outputs[microbatch_i]
    print('outputs', worker.rank, outputs._version)
    
    
    if worker.hyps.use_gpt: 
        raise NotImplementedError()
        
    else: 
        return mlp_model.calc_loss(outputs, targets)

    
def backward_pass(worker, microbatch_i, optimizer, loss):
    
    optimizer.zero_grad()
    
    if worker.last:
        print(worker.rank, 'Backward pass, microbatch %d' % microbatch_i)
        print('loss and inputs versions:', loss._version, worker.inputs[microbatch_i]._version)
        print('loss and inputs grad:', loss.grad, worker.inputs[microbatch_i].grad)
        print('loss and inputs is_leaf:', loss.is_leaf, worker.inputs[microbatch_i].is_leaf)
        
        
        d = loss
        print(d.grad_fn)
        print(d.grad_fn.next_functions)
        print(d.grad_fn.next_functions[0][0].next_functions)
        print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
        #print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
        
        loss.backward()
        print('loss and inputs grad 2:', loss.grad, worker.inputs[microbatch_i].grad)
        
        input_grad = worker.inputs[microbatch_i].grad
        worker.send(input_grad, worker.prev)
        
    if worker.middle:
        output_grad = worker.recv(worker.next)
        print(worker.rank, 'Backward pass, microbatch %d' % microbatch_i)
        worker.outputs[microbatch_i].backward(output_grad)
        input_grad = worker.inputs[microbatch_i].grad
        worker.send(input_grad, worker.prev)
        
        
    if worker.first:
        output_grad = worker.recv(worker.next)
        print(worker.rank, 'Backward pass, microbatch %d' % microbatch_i)
        worker.outputs[microbatch_i].backward(output_grad)
        
    optimizer.step()
    #del worker.inputs[microbatch_i], worker.outputs[microbatch_i]
    
    