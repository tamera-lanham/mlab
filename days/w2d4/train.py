import gin
import random
import torch as t
import torch.nn as nn
import torchtext
from tqdm import tqdm
import transformers

import sys
sys.path.append('/home/ubuntu/mlab/days/w2d2')
import bert_tao


###### Preprocessing

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
    outputs = tokenizer(texts, return_tensors="pt", padding='longest', max_length=max_seq_len, truncation=True)
    return list(zip(sentiments, outputs['input_ids']))

def tokenize(batches, max_seq_len=512):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
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


###### Model

class BertClassifier(nn.Module):
    def __init__(self, pretrained=True, **config):
        super(BertClassifier, self).__init__()
        
        config = {
            "vocab_size": 28996,
            "intermediate_size": 3072,
            "hidden_size": 768,
            "num_classes": 1,
            "num_layers": 12,
            "num_heads": 12,
            "max_position_embeddings": 512,
            "dropout": 0.1,
            "type_vocab_size": 2,
            **config
        }
        
        if pretrained:
            self.bert, _ = bert_tao.my_bert_from_hf_weights(config=config)
        else:
            self.bert = bert_tao.Bert(config)
            
        self.sigmoid = t.nn.Sigmoid()

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        activations = self.sigmoid(outputs.classification)
        return outputs.logits, activations

###### Training


class GetData:
    def __init__(self, batch_size=32):
        data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
        self.data_train_list = list(data_train)
        self.data_test_list = list(data_test)
        
        self.batch_size = batch_size
        
        self.train = preprocess(self.data_train_list, batch_size)
        self.test = preprocess(self.data_test_list, batch_size)
        
    def get_data(self, n_epochs=1, batches_per_epoch=None):
        
        def batch_generator(batched_data, n_epochs):
            for epoch in range(n_epochs):
                for i, batch in enumerate(batched_data):
                    yield i, batch
        
        def batches_tqdm():
            batches = batch_generator(list(zip(self.train, self.test))[:batches_per_epoch], n_epochs)
            return tqdm(batches, total = n_epochs * (batches_per_epoch if batches_per_epoch else len(self.train)))

        return batches_tqdm
    
        


# Current GPU memory usage in GiB
mem = lambda: t.cuda.memory_allocated() / 2**(30) 

# Accuracy score
acc = lambda outputs, targets: sum((outputs > 0.5).squeeze().int() == targets) / len(outputs) 

#@gin.configurable
def train(bert, data_generator, lr=1e-5, device='cuda', val_batches=10):
    
    training_config = {
        'lr': lr,
        'device': device,
        'val_batches': val_batches
    }
    
    device = t.device(training_config['device'])
    
    bert = bert.to(device)
    data = data_generator()
    
    optimizer = t.optim.Adam(bert.parameters(), lr=training_config['lr'])
    loss_func = t.nn.BCELoss()
    
    loss_vals, accs = [], []
    running_loss = 0
    
    for i, (train_batch, test_batch) in data:

        targets, inputs = zip(*train_batch)
        targets = t.Tensor(targets).to(device)
        inputs = t.stack(inputs).to(device)

        optimizer.zero_grad()
        _, outputs = bert(inputs)
        loss = loss_func(outputs, targets[:, None])
        running_loss += loss.detach()
        mem_usage = mem() # moment of highest GPU usage
        loss.backward()
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
                
    
    return loss_vals, accs
    
def save(bert, filename='/home/ubuntu/mlab/days/w2d4/bert-classifier.pt'):
    t.save(bert.state_dict(), filename)
    
