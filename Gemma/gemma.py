
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torchtune.modules import RMSNorm
from tokenizers import Tokenizer
from pathlib import Path
import sentencepiece as spm


@dataclass
class ModelArgs:
    #Hyperparameters

    block_size = 128
    batch_size = 64
    embeddings_dims = 768
    attn_dropout = 0.1
    no_of_heads = 12 #IMP needs to be thoroughly calculated
    dropout = 0.1
    epochs = 100
    max_lr = 2.5e-4
    no_of_decoder_layers = 12 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    device = 'cuda'
    no_kv_heads = 2
    vocab_size = 2000



# spm.SentencePieceTrainer.train('--input=botchan.txt --model_prefix=m --vocab_size=2000')


sp = spm.SentencePieceProcessor()
sp.load('m.model')


#Datasets

# Using tinyshakespeare

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



# Train and test splits
data = torch.tensor(sp.encode_as_ids(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - ModelArgs.block_size, (ModelArgs.batch_size,))
    x = torch.stack([data[i:i+ModelArgs.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ModelArgs.block_size+1] for i in ix])
    x, y = x.to(ModelArgs.device), y.to(ModelArgs.device)
    return x, y



class Normalization(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):  
        super().__init__()
        self.rmsnorm_layer = RMSNorm(dim=embeddings_dims)
        
        
    def forward(self, x):
        
        x = self.rmsnorm_layer(x)
        return x
        
        
        

import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size
    ):
        super().__init__()
        
        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.theta = 0  

    
    def init_matrix(self, seq_len):
        self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), device=ModelArgs.device, requires_grad=False)
        
        positions = torch.arange(seq_len, device=ModelArgs.device).unsqueeze(1)
        theta = 10000 ** (-2 * (positions - 1) / self.embeddings_dims)
        angles = positions * theta
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        indices = torch.arange(self.embeddings_dims, device=ModelArgs.device)

        even_indices = indices[::2]
        odd_indices = indices[1::2]
        
        self.matrix[:, even_indices, even_indices] = cos_angles
        self.matrix[:, odd_indices, odd_indices] = sin_angles
        self.matrix[:, odd_indices, even_indices] = -sin_angles
        self.matrix[:, even_indices, odd_indices] = cos_angles
        
        return self.matrix

    def forward(self, x):
        # B,T,C = x.shape
        # print("MATRIX:",x)
        if(x > self.block_size):
            matrix = self.init_matrix(x)
            return matrix
        else:
            matrix = self.init_matrix(self.block_size)
            
            return matrix


class RotaryAttentionHead(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        no_of_heads: int = ModelArgs.no_of_heads,
        attn_dropout: int = ModelArgs.attn_dropout
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = attn_dropout)
        
    def forward(self,x):
   
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)

        key = self.key(x)
        values = self.value(x)
        matrix = self.rotary_matrix(block_size)
        

        masked = torch.tril(torch.ones((block_size, block_size), device=ModelArgs.device, requires_grad=False))
        rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        weights = rotary_query.permute(2,0,1) @ rotary_key.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
        weights_masked = weights.masked_fill(masked == 0, float('-inf'))
        scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
        scaled_weights = F.softmax(scaled_weights, dim=-1)
        value = scaled_weights @ values
        out = self.dropout(value)
        return out



class MQA(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        no_of_kv_heads: int = ModelArgs.no_of_heads,
        no_of_heads: int = ModelArgs.no_of_heads
    ):
        super().__init__()
        
        self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = no_of_heads // no_of_kv_heads
        self.head_size = embeddings_dims // self.no_of_q_heads
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims)
        # self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device, bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device,  bias=False)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims* self.no_of_q_heads, out_features=embeddings_dims, device=ModelArgs.device, bias=False)
        
        
        
    def scaled_dot_product(self, q, k, v, block_size, matrix):

            masked = torch.tril(torch.ones((block_size, block_size), device=ModelArgs.device, requires_grad=False))
            rotary_query = matrix @ q.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            rotary_key = matrix @ k.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            weights = rotary_query.permute(2,0,1) @ rotary_key.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
            weights_masked = weights.masked_fill(masked == 0, float('-inf'))
            scaled_weights = weights_masked / (torch.sqrt(torch.tensor(k.shape[-1])))
            scaled_weights = F.softmax(scaled_weights, dim=-1)
            value = scaled_weights @ v
            out = self.dropout(value)
            return value
    
    def forward(self,x):
        # print("MQA: ", x.shape)
        batch, block_size, embeddings_dims = x.shape
        multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False) for _ in range(self.no_of_q_heads)])
        # query = self.query(x)
        matrix = self.rotary_matrix(block_size)
            

        key = self.key(x)
        values = self.value(x)

        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), key, values, block_size, matrix) for query in multi_query], dim=-1)
  
        
        linear_layer= self.linear_layer(multi_query_concat)
        out = self.dropout(linear_layer)
        return out
    
    



class GeGLU(nn.Module):
    def __init__(
        self,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)
        self.linear_layer3 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=ModelArgs.device, bias=False)

        
        
        
    def forward(self, x):
        gelu_res = nn.functional.gelu(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(gelu_res, x_V)
        out = self.linear_layer3(res)
        return out
         
         
         
         
class FFN(nn.Module):
    def __init__(self,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                   dropout = ModelArgs.dropout
                 
                 ):
        super().__init__()
        
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=4*embeddings_dims, device=ModelArgs.device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims * 4 , out_features=embeddings_dims, device=ModelArgs.device)
        self.gglu = GeGLU(block_size=block_size, embeddings_dims=embeddings_dims* 4)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.linear_layer1(x)
        x = self.gglu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        return x
    
    
    

class DecoderLayer(nn.Module):
    def __init__(self, 
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,
                 
                 ) :
        super().__init__()
        
        
        self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size)
        self.mqa = MQA(embeddings_dims=embeddings_dims, block_size=block_size, no_of_kv_heads=ModelArgs.no_kv_heads, no_of_heads=ModelArgs.no_of_heads)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        
        x = self.norm(x + self.mqa(x))
        x = self.norm(x + self.feedforward_network(x))
        return x
    
    
class Gemma(nn.Module):
    def __init__(self, 
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                  dropout = ModelArgs.dropout
                 
                 ) :
        super().__init__()
        
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims, device=ModelArgs.device)
        self.decoder = nn.Sequential(*[DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size, device=ModelArgs.device)
        self.dropout = nn.Dropout(p = dropout)
        self.norm = Normalization(embeddings_dims)
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        # x = self.norm(x)
        x = self.linear_layer(x)
        # out = self.norm(x)
        return x
    
    


# Instantiating the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
ModelArgs.device = device
model = Gemma(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
model = model.to(ModelArgs.device)



#Printing a summary of the architecture
from torchinfo import summary
idx, targets = get_batch('test')
# idx = idx.to(device)
summary(model=model,
        input_data=idx,
        # input_size=(ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.embeddings_dims),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])


# Optimizer setup and scheduler steup

optimizer = torch.optim.AdamW(params=model.parameters(), lr=ModelArgs.max_lr)

total_steps = 5000
eval_iters = 100
# lr_scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= total_steps - initial_iters)

@torch.inference_mode()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            idx, targets = get_batch(split=split)
            logits = model(idx)
            batch_size, block_size, embeddings_dims = logits.shape
            logits = logits.view(batch_size*block_size, embeddings_dims) # Total tokens(words) => batch_size * block_size
            targets = targets.view(batch_size * block_size)
            loss = nn.functional.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



#Train the  model
from tqdm import tqdm

model.train()
for step in tqdm(range(total_steps)):

    # every once in a while evaluate the loss on train and val sets
    if (step  % eval_iters == 0 and step != 0) or step == total_steps - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # torch.save(model.state_dict(), 'weights/Gemma1_32M_steps_%d.pth' % (step))

    idx, targets = get_batch(split='train')
    logits = model(idx)
    batch_size, block_size, embeddings_dims = logits.shape
    logits = logits.view(batch_size*block_size, embeddings_dims)
    targets = targets.view(batch_size * block_size)
    loss = nn.functional.cross_entropy(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
