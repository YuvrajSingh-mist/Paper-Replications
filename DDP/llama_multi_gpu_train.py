
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torchtune.modules import RMSNorm
from tokenizers import Tokenizer
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from torch.utils.data import Dataset, DataLoader
import wandb 
from tqdm import tqdm 

import os

def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost' 
    # os.environ['MASTER_PORT'] = '12355'  
    init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def cleanup():
    destroy_process_group()



@dataclass
class ModelArgs:
    #Hyperparameters

    block_size = 128
    batch_size = 64
    embeddings_dims = 384
    attn_dropout = 0.1
    no_of_heads = 6 #IMP needs to be thoroughly calculated
    dropout = 0.1
    # epochs = 100
    max_lr = 1e-4
    no_of_decoder_layers = 6 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    clip = 1.0
    device = 'cuda:0'
    no_kv_heads = 2
    vocab_size = 10000


from pathlib import Path
data_path = Path('data')
data_path.mkdir(exist_ok=True)
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# !cp input.txt data/input.txt



#Datasets

# Using tinyshakespeare

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



def save_checkpoint(model):
    ckp = model.module.state_dict()
    torch.save(ckp, "checkpoint.pt")
    print("Checkpoint saved")



#Subword level tokenization

#Loading custom trained BPE
# Load the tokenizer
# tokenizer = Tokenizer.from_file("data/bpe_tokenizer_tinyshakespeare_1k.json")
# vocab_size = tokenizer.get_vocab_size()
# Encode and decode functions
# encode = lambda s: tokenizer.encode(s).ids
# decode = lambda l: tokenizer.decode(l)





###############################################################################
#Character level tokenization

# # here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)


# create a mapping from characters to integers
stoi = { ch: i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
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


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y


encoded_data = torch.tensor(encode(text), dtype=torch.long)

train_dataset = TextDataset(train_data, ModelArgs.block_size)
val_dataset = TextDataset(val_data, ModelArgs.block_size)




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
        


# import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size
    ):
        super().__init__()
        
        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.theta = 0  

        
    # def init_matrix(self, seq_len):
    #         self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False)
    #         for pos in range(seq_len):
    #             for j in range(1, self.embeddings_dims // 2):
    #                 self.theta = 10000 ** (-2*(pos-1) / self.embeddings_dims)
    #                 self.matrix[pos, 2*j + 1, 2*j + 1] = np.cos((pos*self.theta))
    #                 self.matrix[pos, 2*j + 1, j + 1] = -np.sin((pos* self.theta))
    #                 self.matrix[pos, 2*j , 2*j ] = -np.cos((pos* self.theta))
    #                 self.matrix[pos, 2*j + 1, 2*j + 1] = np.sin((pos* self.theta))
    #         return self.matrix
        self.device=device
        
    def init_matrix(self, seq_len):
        self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)
        
        positions = torch.arange(seq_len,  dtype=torch.float32,  device = self.device).unsqueeze(1)
        # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        theta = 10000 ** (-2 * (positions - 1) / self.embeddings_dims)
        angles = positions * theta
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        indices = torch.arange(self.embeddings_dims,  dtype=torch.int64,  device = self.device)
        # print(indices)
        # print(indices.shape)
        # print(indices[::2])
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
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        no_of_heads: int = ModelArgs.no_of_heads,
        attn_dropout: int = ModelArgs.attn_dropout
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
    def forward(self,x):
        # print(x.shape)
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)
        # print(query)
        key = self.key(x)
        values = self.value(x)
        matrix = self.rotary_matrix(block_size)
        
        # print(matrix.shape)
        # print(query.shape)
        masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
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
        device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        no_of_kv_heads: int = ModelArgs.no_of_heads,
        no_of_heads: int = ModelArgs.no_of_heads,
       
    ):
        super().__init__()
        
        self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = no_of_heads // no_of_kv_heads
        self.head_size = embeddings_dims // self.no_of_q_heads
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
        # self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.device = device
        self.multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False,  device = self.device) for _ in range(self.no_of_q_heads)])
        
    def scaled_dot_product(self, q, k, v, block_size, matrix):
            
            # masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))

            masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
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
        
        # query = self.query(x)
        matrix = self.rotary_matrix(block_size)
            

        key = self.key(x)
        values = self.value(x)

        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), key, values, block_size, matrix) for query in self.multi_query], dim=-1)
  
        
        linear_layer= self.linear_layer(multi_query_concat)
        out = self.dropout(linear_layer)
        return out


class GQA(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        no_of_q_heads: int = ModelArgs.no_of_heads,
        no_of_kv_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()
        
        self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = no_of_q_heads
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_kv_heads, out_features=embeddings_dims , dtype=torch.float32,  bias=False,  device = device)
        self.device = device
        self.mqa = nn.ModuleList([MQA(embeddings_dims=embeddings_dims, device = self.device, block_size=block_size) for _ in range(self.no_of_kv_heads)])
        
    def forward(self,x):
        
        batch, block_size, embeddings_dims = x.shape
        

        grouped_query_concat = torch.cat([group(x) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat)
        out = self.dropout(linear_layer)
        return out


class Swish(nn.Module):
    def __init__(
        self,
         device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        
        self.sig = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        swish = x * self.sig(x)
        
        return swish
         


class SWiGLU(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)

        
        
        
    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out
         


class FFN(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                   dropout = ModelArgs.dropout
                 
                 ):
        super().__init__()
        
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        
        x = self.swiglue(x)
        x = self.linear_layer(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, 
                  device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,
                 
                 ) :
        super().__init__()
        
        
        self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, no_of_kv_heads=ModelArgs.no_kv_heads, no_of_q_heads=ModelArgs.no_of_heads,  device = device)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.norm2 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):
        
        x = self.norm1(x + self.gqa(x))
        x = self.norm2(x + self.feedforward_network(x))
        return x


class Llama(nn.Module):
    def __init__(self, 
                device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                  dropout = ModelArgs.dropout
                 
                 ) :
        super().__init__()
        
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.Sequential(*[DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers)])
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        # self.norm = Normalization(embeddings_dims)
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        # x = self.norm(x)
        x = self.linear_layer(x)
        # out = self.norm(x)
        return x


# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# ModelArgs.device = device
model = Llama(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
model = model.to(ModelArgs.device)

#Printing a summary of the architecture
# !pip install torchinfo 
from torchinfo import summary
idx, targets = get_batch('test')
# sample_idx = random.randint(range(len(train_dataset)))
# idx, targets = train_dataset[0]
idx = idx.to(ModelArgs.device)
# targets = targets.to(ModelArgs.device)
summary(model=model,
        input_data=idx,
        # input_size=(ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.embeddings_dims),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# %% [code]

def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused



#Train the  model

def train():
    setup()
    device=torch.distributed.get_rank()
    
    
    # rank = torch.distributed.get_rank()
    print(f"Start running basic DDP example on rank {device}.")
    # # create model and move it to GPU with id rank
    # device_id = rank % torch.cuda.device_count()
    # CFG = ModelArgs()
    
    if(device == 0):
        # Initialise run
        wandb.init(
            # entity = 'rajceo2031',
                        project = 'Llama-DDP',
                        # config = CFG,
                        # save_code = True,
                        #group = 'ANN',
                        #job_type = 'train'
)
    
    model = Llama(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=device)
    # Optimizer setup and scheduler steup
    torch.cuda.set_device(device)
    model = model.cuda()
        
    # Wrap model with DDP after moving to GPU
    model = DDP(model, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=ModelArgs.max_lr)
    val_loader = DataLoader(val_dataset, batch_size=ModelArgs.batch_size, shuffle=False, sampler = DistributedSampler(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=ModelArgs.batch_size, shuffle=False, sampler = DistributedSampler(train_dataset))
    
    
   
        
    save_chechpoint_iter = 100
    total_iters = 25000
    eval_iters = 100
    # for X,y in train_loader:
    #     print(X.shape)
    #     print(y.shape)

     # Only create progress bar for rank 0
    # eval_epoch_iterator = range(eval_iters)
    train_epoch_iterator = range(total_iters)
    if device == 0:
        train_epoch_iterator = tqdm(train_epoch_iterator, desc="Training")
        # eval_epoch_iterator = tqdm(eval_epoch_iterator, desc='Validation')
   
    # lr_scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= total_steps - initial_iters)

    @torch.inference_mode()
    def estimate_loss():
        out = {}
        
        model.eval()
        loader = None
        # print("Starting the eval...")
        for split in ['train', 'val']:
            print(f"Starting with {split} evaluation...")
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                # idx, targets = get_batch(split=split)
                if(split == 'train'):
                    loader = train_loader
                else:
                    loader = val_loader
                    
                # for idx, targets in loader:
                idx, targets = next(iter(loader))
                idx = idx.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                logits = model(idx)
                batch_size, block_size, embeddings_dims = logits.shape
                logits = logits.view(batch_size*block_size, embeddings_dims) # Total tokens(words) => batch_size * block_size
                targets = targets.view(batch_size * block_size)
                loss = nn.functional.cross_entropy(logits, targets)
                losses[k] = loss.item()

                # if device == 0:
                #     eval_epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                    
            out[split] = losses.mean()
            

        model.train()
        return out

    # model = model.to(rank)
    model.train()
    
    # for step in tqdm(range(total_iters)):
    for step in train_epoch_iterator:
        train_loader.sampler.set_epoch(step)
        val_loader.sampler.set_epoch(step)

        # every once in a while evaluate the loss on train and val sets
        if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
            losses = estimate_loss()
            # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if device == 0:  # Only print on main process
                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # Log training loss more frequently
        # if device == 0:
                wandb.log({
                    "training_step_loss": losses['train'],
                    "val_step_loss": losses['val'],
                    "step": step
                })
        if(step % save_chechpoint_iter == 0 and device == 0 and step != 0):
            print(f"Saving the model checkpoint for step: {step}")
            save_checkpoint(model)
        
        
       
        # idx, targets = get_batch(split='train')
        # print(f"Starting the train step: {step}...")
        # for idx, targets in train_loader:
        idx, targets = next(iter(train_loader))
        idx = idx.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        logits = model(idx)
        batch_size, block_size, embeddings_dims = logits.shape
        logits = logits.view(batch_size*block_size, embeddings_dims)
        targets = targets.view(batch_size * block_size)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # if device == 0:
        #     train_epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        # print(loss.item())
        # break

        # if step != 0 and (step % eval_iters == 0 or step == total_steps -1) :
        #     loss_values = estimate_loss()
        #     print("Train Loss at {} steps : {}".format(step, loss.item()), "Val Loss at {} steps : {}".format(step, loss_values['val']))

        # Add after a training step:
        # unused_params = find_unused_parameters(model)
        # print("Unused parameters:", unused_params)
        # break
        

    # Cleanup
    if device == 0:
        wandb.finish()
    cleanup()

    
world_size = torch.cuda.device_count()
print(f"World size: {world_size}")
train()

