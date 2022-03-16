import transformers
import torch as t

class BlockWrapper(t.nn.Module):
    def __init__(self, gpt_block):
        super().__init__()
        self.model = gpt_block
    
    def forward(self, inputs):
        activations, *_ = self.model(inputs)
        return activations
    
def split_model(model, n_gpus):
    n_blocks = len(model.transformer.h)
    starts = t.linspace(0, n_blocks, n_gpus + 1).int()[:-1] # Starting index of each section
    ends = t.linspace(0, n_blocks, n_gpus + 1).int()[1:]
    blocks = [BlockWrapper(block) for block in model.transformer.h]
    gpt_block_sections = [t.nn.Sequential(*blocks[start:end]) for start, end in zip(starts, ends)]

    first = t.nn.Sequential(
        model.transformer.wte,
        model.transformer.drop,
        gpt_block_sections[0]
    )

    last = t.nn.Sequential(
        gpt_block_sections[-1],
        model.transformer.ln_f,
        model.score
    )

    models = [first] + gpt_block_sections[1:-1] + [last]
    return models

filename_pattern = 'gpt-j-%d.pt'

def split_model_and_save(n_parts = None, model = None, filename_pattern = filename_pattern):
    
    if not model: 
        model = transformers.AutoModelForSequenceClassification.from_pretrained("EleutherAI/gpt-j-6B")
        
    models = split_model(model, n_parts)
    for i, model in enumerate(models):
        t.save(model, filename_pattern % i)

def load_model(rank, filename_pattern=filename_pattern):
    return t.load(filename_pattern % rank)
