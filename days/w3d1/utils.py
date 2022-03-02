import random
import torchtext
from tqdm import tqdm
import transformers
import pdb
import torch as t

class BlockWrapper(t.nn.Module):
    def __init__(self, gpt_block):
        super().__init__()
        self.model = gpt_block
    
    def forward(self, inputs):
        activations, *_ = self.model(inputs)
        return activations

def batch(data, batch_size):
    batches, batch = [], []
    for i, sample in enumerate(data, 1):
        if i % batch_size == 0:
            batches.append(batch)
            batch = []
        batch.append(sample)

    batches.append(batch)
    return batches

def tokenize_batch(batch, tokenizer, max_seq_len):
    sentiments, texts = zip(*batch)
    outputs = tokenizer(list(texts), return_tensors="pt", padding='longest', max_length=max_seq_len, truncation=True)
    return list(zip(sentiments, outputs['input_ids']))

def tokenize(batches, max_seq_len=512):
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    return [tokenize_batch(batch, tokenizer, max_seq_len) for batch in tqdm(batches)]

def convert_to_int(batches):
    conv_dict = {
        "pos": 1, 
        "neg": 0
    }
    return [
            [(conv_dict[sentiment], text) for sentiment,text in batch] 
            for batch in batches
            ]

def preprocess(data, batch_size, max_seq_len=512):
    
    batched_data = batch(data, batch_size)
    random.shuffle(batched_data)
    tokenized = tokenize(batched_data, max_seq_len)
    preprocessed = convert_to_int(tokenized)
    
    return preprocessed

def imdb_data(batch_size=32, max_seq_len=512):
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    
    data_train_list = list(data_train)
    data_test_list = list(data_test)

    tokenized_train_batches = preprocess(data_train_list, batch_size, max_seq_len)
    tokenized_test_batches = preprocess(data_test_list, batch_size, max_seq_len)

    return tokenized_train_batches, tokenized_test_batches

def fake_imdb_data(batch_size=32, max_seq_len=512, n_batches=10, vocab_size=50257):
    
    def sample():
        sentiment = int(t.randint(0, 2, (1,)))
        tokens = t.randint(0, vocab_size, (max_seq_len,)).long()
        return sentiment, tokens
    
    train_batches = [[sample() for _ in range(batch_size)] for _ in range(n_batches)]
    test_batches = [[sample() for _ in range(batch_size)] for _ in range(n_batches)]
    
    return train_batches, test_batches
    
    
def batch_to_microbatches(batch, n_microbatches, microbatch_size): # batch is list of [t.tensor()] 
    if not len(batch) ==  n_microbatches * microbatch_size: raise ValueError('Bad batch size :(')
    return [t.stack(batch[i*microbatch_size:(i+1)*microbatch_size]) for i in range(n_microbatches)]
            
            
mem = lambda rank: t.cuda.memory_allocated('cuda:%d' % rank) / 2**(30) 
