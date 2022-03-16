import torch as t

def n_gpus():
    return t.cuda.device_count()

