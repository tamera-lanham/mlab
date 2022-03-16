# Stuff going on in the code

Multiprocessing stuff
- Processes / groups
- Distributing tensors

Data
- Preprocessing
- Data generator?
- Some things are lists, some are tensors 
- Inputs belong to rank=0, outputs and n_tokens belong to last
- Turning batches into microbatches

Training
- Forward and backward passes
- Hyperparams:
    - Optimizer choice
    - learning rate
    - n epochs
    - batch size and microbatch size
    - other optimizer params
    
    
# Things I want to refactor

- Break stuff into files
- Get all the distributed stuff packed away
- Turn all the lists to tensors