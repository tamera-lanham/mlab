import torch as t
import torch.nn as nn
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, layer_sizes=[16, 128, 128, 16], bias=False, device=None):
        super().__init__()
        layers = [
            layer for i, size in enumerate(layer_sizes[:-1]) 
            for layer in (nn.Linear(size, layer_sizes[i+1], bias, device), nn.ReLU())]
        self.layers = nn.Sequential(*layers[:-1])
    
    def forward(self, x):
        return self.layers(x)
    
def split_MLP(model, n_parts):
    # All blocks are guaranteed to start with a Linear layer
    
    n_lin_layers = len([l for l in model.layers if isinstance(l, nn.Linear)])
    assert n_parts <= n_lin_layers
    
    starts = 2*t.linspace(0, n_lin_layers, n_parts + 1).int()[:-1]
    ends = 2*t.linspace(0, n_lin_layers, n_parts + 1).int()[1:]
    
    models = [nn.Sequential(*model.layers[s:e]) for s, e in zip(starts, ends)]
    
    return models

filename_pattern = 'mlp-%d.pt'

def split_MLP_and_save(model, n_parts, filename_pattern=filename_pattern):
    models = split_MLP(model, n_parts)
    for i, model in enumerate(models):
        t.save(model, filename_pattern % i)

def load_model(rank, filename_pattern=filename_pattern):
    return t.load(filename_pattern % rank)

def calc_loss(outputs, targets):
    loss_fn = t.nn.MSELoss()
    return loss_fn(outputs.float(), targets.float())
    