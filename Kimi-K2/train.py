

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm 
from dataclasses import dataclass
from torch.nn import RMSNorm
# from tokenizers import Tokenizer
from pathlib import Path
import os

import wandb
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# Load model directly
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token='...')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#liger kernels
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


from muon import MuonWithAuxAdam




# torch.autograd.set_detect_anomaly(True)

def setup_ddp():
    """Initialize DDP setup"""
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Get local rank from environment variable set by torchrun
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return local_rank, world_size, rank, device

def cleanup_ddp():
    """Clean up DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class ModelArgs:
    #Hyperparameters

    block_size = 128 
    batch_size = 256
    embeddings_dims = 384
    attn_dropout = 0.1
    no_of_heads = 8 #IMP needs to be thoroughly calculated
    dropout = 0.1
    epochs = 1
    max_lr = 6e-4
    no_of_decoder_layers = 6 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    device = 'cuda'
    vocab_size = len(tokenizer.get_vocab())
    base_freq=100000
    # s = 1.0
    experts=8
    clip = 1.0
    top_experts=2
    noisy_topk = False
    use_checkpointing = False
    use_liger = True  # Use Liger kernels for optimized operations
    use_shared_expert = True  # Enable/disable shared expert
    ignore_pad_token_in_loss = True  # Whether to ignore padding tokens in loss calculation
    eps: float = 1e-8
    loss_scale = 0.3
    useauxFreeLoadBalancingLoss = True  
    aux_free_bias_update_rate = 0.001
    latent_dim = 64  # Latent dimension for attention
#Datasets

# Using tinyshakespeare

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



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


# Create a mapping from characters to integers
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


tinystories = True
fw = False
fw_train = None
fw_test = None
if(tinystories):
    
    fw_train = load_dataset("roneneldan/TinyStories", split="train")
    fw_test = load_dataset("roneneldan/TinyStories", split="validation")
    print(fw_train)
    print(fw_test)
if(fw):   
    fw_train = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
    fw_train = fw_train.train_test_split(test_size=0.01)
    print(fw_train)
    print(fw_train)




def prepare_dataset(split, device, batch_size, use_ddp=False):
    print("Device is: ", device)
 
    def collate_fn(batch):
        # Extract text data
        texts = []
        
        for item in batch:
            tt = item['text']# Append EOS token to each text
            texts.append(tt)

        input_encodings = tokenizer(texts, max_length = ModelArgs.block_size,padding='max_length', truncation=True, return_tensors="pt")
        
        input_encodings["labels"] = input_encodings["input_ids"].clone()  # Use `input_ids` as labels

        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]  
        # input_encodings['labels'][:, 0] = tokenizer.bos_token_id
        # input_encodings["labels"][:, -1] = tokenizer.eos_token_id  # Let the last token be end 
       
        return input_encodings

  
    dataloader = None
    if(tinystories):
        if(split == 'train'):
            sampler = DistributedSampler(fw_train, shuffle=True) if use_ddp else None
            shuffle = False if use_ddp else True
            data_loader = DataLoader(
            fw_train,
            # generator=generator,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
        )
        elif(split == 'val'):
            sampler = DistributedSampler(fw_test, shuffle=False) if use_ddp else None
            shuffle = False if use_ddp else False
            data_loader = DataLoader(
            fw_test,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
        )
    elif(fw):
        if(split == 'train'):
            sampler = DistributedSampler(fw_train['train'], shuffle=True) if use_ddp else None
            shuffle = False if use_ddp else True
            data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
    )
        elif(split == 'val'):
            sampler = DistributedSampler(fw_train['test'], shuffle=False) if use_ddp else None
            shuffle = False if use_ddp else False
            data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
        )
    return data_loader





    
    

# from andrej karapathy github
def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    
    if(len(input_ids[0]) < max_length):
        max_length -= len(input_ids[0]) # If the input is longer than max_length, set max_length to the length of the input
    else:
        max_length = len(input_ids[0]) - max_length  
    for _ in range(max_length):
        with torch.no_grad(), torch.autocast(device_type=ModelArgs.device, dtype=torch.bfloat16):
            # Pass inference=True to use the inference path in the model
            outputs = model(input_ids, inference=True)
            logits = outputs[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            input_ids = torch.cat([input_ids, xcol], dim=1) #1 because is it the dimension of the sequence
            
            if xcol.item() == tokenizer.eos_token_id:
                break
            
            
    return tokenizer.decode(input_ids[0])



class Normalization(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):  
        super().__init__()
        self.rmsnorm_layer = RMSNorm(embeddings_dims)
        
        
    def forward(self, x):
        
        x = self.rmsnorm_layer(x)
        return x
        


class Swish(nn.Module):
    def __init__(
        self,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        device = ModelArgs.device
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        swish = x * self.sig(x)

        return swish



class SWiGLUExpertMoE(nn.Module):
    def __init__(
        self,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        device = ModelArgs.device
    ):
        super().__init__()

        self.hidden_dims = (embeddings_dims * 2)
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
        self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, device = device)




    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out



class MoeLayer(nn.Module):
    def __init__(
        self,
        dropout = ModelArgs.dropout,
        embeddings_size = ModelArgs.embeddings_dims,
        device = ModelArgs.device,
        # inner_dimensional_states: int = 3072
    ):
        super().__init__()

        self.heads = nn.ModuleList([SWiGLUExpertMoE() for _ in range(ModelArgs.experts)])
        self.gate = nn.Linear(in_features=embeddings_size, out_features=ModelArgs.experts, device=device, bias=False)
        
        # Only create shared expert if enabled
        if ModelArgs.use_shared_expert:
            self.shared_expert = SWiGLUExpertMoE()
        else:
            self.shared_expert = None
            
        if(ModelArgs.noisy_topk is True and ModelArgs.use_checkpointing == False):
            self.noise = nn.Linear(in_features=embeddings_size, out_features=ModelArgs.experts, device=device, bias=False)
            self.noisy_router = None
        # self.outputs = torch.zeros((batch_size,block_size, embeddings_size), device=device) #batch size needs to be defined because we are accessing it explicitly
        self.device = device
        # self.shared_expert_out = torch.zeros((ModelArgs.batch_size, ModelArgs.embeddings_dims), device=device)
        # self.b = torch.zeros((ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.experts), device=device)

        if ModelArgs.useauxFreeLoadBalancingLoss:
            self.register_buffer('routing_bias', torch.zeros(ModelArgs.experts, device=self.device))
            # self.routing_bias = torch.zeros(ModelArgs.experts, device=self.device)
            self.bias_update_speed = ModelArgs.aux_free_bias_update_rate
        
        
    def forward(self, x):
        # mlp_weights_init = self.mlp.apply(weights_init)
        self.gate_out = self.gate(x) #[bz, seq, num_experts]

        
        if(ModelArgs.noisy_topk == True and ModelArgs.use_checkpointing == False):
            noise = self.noise(x)
            gaussian_noise = torch.normal(0, 1, size=self.gate_out.shape, device=self.device)
            self.noisy_router = F.softplus(noise) * gaussian_noise
            self.gate_out += self.noisy_router
            
        

        shared_output = 0
        out = 0

        

        if ModelArgs.useauxFreeLoadBalancingLoss:
            
           self.gate_out += self.routing_bias
           
        
                
        
        # Adjust top_k based on whether shared expert is used
        top_k = ModelArgs.top_experts
        top_k_values, top_k_indices = torch.topk(self.gate_out, k=top_k) #[bs, seq len, top k]
        # topkmask = torch.ones_like(top_k_values, device=self.device)  # [bs, seq len, experts]
        # indices = torch.arange(top_k_values.size(0), device=self.device).unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]
        # topkvaluesMasked = top_k_values.masked_fill(indices != top_k_indices, float('-inf'))  # Mask out negative values
        masked = torch.full_like(self.gate_out, float('-1e20'), device=self.device) 
        masked_values = masked.scatter_(-1, top_k_indices, top_k_values)
        probs = torch.nn.functional.softmax(masked_values, dim=-1) #[bs, seq len, top k]
        
        out = torch.zeros_like(x)
        if ModelArgs.use_shared_expert and self.shared_expert is not None:
            shared_output += self.shared_expert(x)

        flat_x = x.view(-1, x.size(-1))  # Flatten the input for easier processing

        for i in range(ModelArgs.experts): # Iterate through each expert index (0 to num_experts-1)
            # Determine which tokens routed to this expert 'i'
            # top_k_indices is [bs, seq_len, self.top_k]
            # We want a mask of shape [bs, seq_len] where True if expert 'i' is in the top_k for that token
            expert_i_is_chosen_mask = (top_k_indices == i).any(dim=-1) # Check along the top_k dimension
            # expert_i_is_chosen_mask has shape [bs, seq_len]

            if not expert_i_is_chosen_mask.any(): # If expert 'i' was not chosen by any token
                continue

            # Flatten the mask to apply to flat_x
            flat_expert_i_is_chosen_mask = expert_i_is_chosen_mask.reshape(-1) # Shape: [bs * seq_len]

            # Select input tokens for this expert
            selected_input_tokens = flat_x[flat_expert_i_is_chosen_mask] # Shape: [num_active_for_expert_i, embed_dim]

            if selected_input_tokens.numel() == 0: # Should be caught by .any() above, but good check
                continue

            # Process through the expert
            expert_output_for_selected = self.heads[i](selected_input_tokens)

            # Get the routing probabilities for these chosen tokens specifically for expert 'i'
            # routing_probs is [bs, seq_len, num_experts]
            # expert_i_probs_original_shape = routing_probs[:, :, i] # Probabilities for expert 'i', shape [bs, seq_len]
            # flat_expert_i_probs = expert_i_probs_original_shape.reshape(-1) # Shape [bs * seq_len]
            # active_token_weights = flat_expert_i_probs[flat_expert_i_is_chosen_mask] # Shape: [num_active_for_expert_i]

            # Alternative way to get weights directly using the mask on routing_probs for expert i:
            # Get the [bs, seq_len] slice of probabilities for the current expert 'i'
            probs_for_expert_i = probs[:, :, i] # Shape: [bs, seq_len]
            # Now use the expert_i_is_chosen_mask (which is also [bs, seq_len]) to select the relevant weights
            active_token_weights = probs_for_expert_i[expert_i_is_chosen_mask] # Shape: [num_active_for_expert_i]


            weighted_expert_output = expert_output_for_selected * active_token_weights.unsqueeze(-1)

            # Add this expert's contribution
            temp_contribution_for_expert_i = torch.zeros_like(x) # Initialize with zeros
            temp_contribution_for_expert_i.masked_scatter_(
                expert_i_is_chosen_mask.unsqueeze(-1).expand_as(x), # Use the original 2D mask, expanded
                weighted_expert_output
            )
            out = out + temp_contribution_for_expert_i
            
            
        # for expert_idx in range(ModelArgs.experts):
        #     # Create mask for current expert across all top_k positions
        #     expert_mask = (top_k_indices == expert_idx)

        #     # Sum probabilities for current expert
        #     expert_weights = (probs * expert_mask).sum(dim=-1)  # [batch, seq_len]

        #     # Get inputs where expert is used
        #     selected = expert_weights > 0
        #     if not selected.any():
        #         continue
        #     # print(expert_weights.shape)
        #     # print(x[selected].shape)

        #     # Process all selected inputs through expert
        #     expert_out = self.heads[expert_idx](x[selected])
            
            
                
        #     # Weight and accumulate outputs
        #     out[selected] += expert_out * expert_weights[selected].unsqueeze(-1)

        out = out + shared_output  # Add shared expert output if enabled
        
        if ModelArgs.useauxFreeLoadBalancingLoss and self.training:
            
            with torch.no_grad():  
                ci = probs.sum(dim=(0,1))  # Su  of tokens for each expert
                ci_avg = ci.mean()
                
                
                error_i = ci_avg - ci
                
                self.update = self.bias_update_speed * torch.sign(error_i)  # Update routing bias
                self.routing_bias.add_(self.update)
                # self.routing_bias = self.routing_bias + self.update

        return out
    
    
# import numpy as np
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size,
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # Create positional encoding matrix
        pe = torch.zeros(block_size, embeddings_dims)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embeddings_dims, 2).float() * (-math.log(10000.0) / embeddings_dims))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it's not a parameter but moves with the model
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, block_size, embeddings_dims]

    def forward(self, x):
        # x shape: [batch_size, seq_len, embeddings_dims]
        batch_size, seq_len, _ = x.shape
        
        # Add positional embeddings
        # pe[:, :seq_len] ensures we only use the positional embeddings up to the sequence length
        pos_emb = self.pe[:, :seq_len].to(x.device)
        return pos_emb



class LatentAttention(nn.Module):
    def __init__(
        self,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
        device = ModelArgs.device
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.no_of_heads = no_of_heads
        # if(ModelArgs.use_flash_attention==False):
        self.latent_dim = ModelArgs.latent_dim
        self.W_k = nn.Linear(in_features=self.latent_dim, out_features=self.head_size, device=device, bias=False)
        self.W_v = nn.Linear(in_features=self.latent_dim, out_features=self.head_size, device=device, bias=False)
        self.W_dkv = nn.Linear(in_features=ModelArgs.embeddings_dims, out_features=self.latent_dim, device=device, bias=False) # 3 for query, key and value
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device, bias=False)
        # self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=ModelArgs.device, bias=False)
        # self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=ModelArgs.device,bias=False)
    # self.dropout = nn.Dropout(p = attn_dropout)
        

        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device

        # Use sinusoidal positional embeddings instead of rotary
        self.pos_embeddings = SinusoidalPositionalEmbeddings(embeddings_dims=self.head_size, device=device)
        # self.register_buffer('absorbed_q', None)
        # self.absorbed_q = None
        
    def forward(self, x, kv_cache=None, mask=None):
        batch_size, block_size, embd_dims = x.shape

        # k = self.keys(x)
        # q = self.query(x)
        # v = self.values(x)
        
        self.latent_matrix = self.W_dkv(x)

        # print("q shape: ", q.shape)
        
        # print("Shape of latent mat: ", self.query.weight.shape)
        # print("Shape of compressed_k: ", self.W_k.weight.shape)
        
        # if(self.absorbed_q is None):
        self.absorbed_q = torch.matmul(self.query.weight.T , self.W_k.weight)
        
        
        # weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)

        # if kv_cache is None:
        if kv_cache is None:
            kv_cache = self.latent_matrix
        else:
            # print(kv_cache)
            # print("Shape of latent matrix: ", self.latent_matrix.shape)
            # print("Shape of kv_cache: ", kv_cache.shape)
            kv_cache = torch.cat([kv_cache, self.latent_matrix], dim=1)

        self.compressed_k = self.W_k(kv_cache)
        self.compressed_v = self.W_v(kv_cache)
        
        q_res = torch.matmul(x , self.absorbed_q)
        weights =  q_res @ torch.transpose(kv_cache, dim0=-2, dim1=-1) * (self.head_size ** -0.5)  # [batch_size, block_size, block_size]
        # print("Shape of weights: ", weights.shape)
        # print("Shape of kv_cache: ", kv_cache.shape)
        if(mask is not None):
            weights = weights.masked_fill(mask == 0, float('-1e20')) #Masking the attention weights
            
        masked_table = torch.tril(torch.ones(q_res.shape[1], kv_cache.shape[1], device=ModelArgs.device))

        masked_values = weights.masked_fill(masked_table[: q_res.shape[1], : kv_cache.shape[1]] == 0, float('-1e20'))
        weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
        weights_normalized = self.dropout(weights_normalized)
        
        # print("Shape of weights_normalized: ", weights_normalized.shape)
        # Apply positional embeddings to the output
        
        
        
        
        # print("Shape of compressed_v: ", self.compressed_v.shape)
        out = weights_normalized @ self.compressed_v
        
        # out = self.pos_embeddings(out)
        return out, kv_cache

# MHA


class MHLA(nn.Module):
    def __init__(
        self,
        device,
        attn_dropout = ModelArgs.attn_dropout,
        embeddings_dims = ModelArgs.embeddings_dims,
        no_of_heads = ModelArgs.no_of_heads,
    ):
        super().__init__()
        self.heads = nn.ModuleList([LatentAttention(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings
        
    def forward(self, x, kv_cache=None, mask=None):
        # concat = torch.cat([head(x, kv_cache=kv_cache, mask=mask) for head in self.heads], dim=-1)
        res = []
        for head in self.heads:
            head_out, kv_cache = head(x, kv_cache=kv_cache, mask=mask)
            res.append(head_out)
        concat = torch.cat(res, dim=-1)  # Concatenate along the last dimension
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out, kv_cache

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
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, device = device)

        self.dropout = nn.Dropout(p = dropout)  # Uncommenting the dropout line
    def forward(self, x):

        x = self.linear_layer(x)
        x = F.gelu(x)
        x = self.linear_layer2(x)
        x = F.gelu(x)
        # x = self.dropout(x)  # Uncommenting the dropout line
        return x







class DecoderLayer(nn.Module):
    def __init__(self,
                device,
                attn_dropout: float = ModelArgs.attn_dropout,
                no_of_heads: int = ModelArgs.no_of_heads,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,

                 ) :
        super().__init__()

        # self.base_freq = ModelArgs.base_freq
        # self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.mha = MHLA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, device=device)
        self.layer_norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.layer_norm2 = Normalization(embeddings_dims=embeddings_dims)
        # self.layer_norm3 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)

        self.moe_block = MoeLayer(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, x, kv_cache=None, ffn=None, mask=None):

        out, kv_cache = self.mha(self.layer_norm1(x), kv_cache=kv_cache, mask=mask)  #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = x + out  # Fixed: removed in-place operation
        x = x + self.moe_block(self.layer_norm2(x)) #Very important step

        return x, kv_cache


class Block(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                  dropout = ModelArgs.dropout

                 ) :
        super().__init__()
        self.base_freq = ModelArgs.base_freq
        # self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.ModuleList(DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers))
        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        self.norm = Normalization(embeddings_dims)
        
        #weight tying
        # self.embeddings.weight = self.linear_layer.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, x, mask=None, actual_labels = None, inference=False):
        index = 0
        no_of_layers = 0
        # x = self.embeddings(x)
        # # x = self.dropout(x)
        # if(mask is not None):
        kv_cache = None
        #     x = x * mask
        #     # mask = mask.unsqueeze(-1)
        # x = self.decoder(x)
        for layer in self.decoder:
            # if no_of_layers % 2 == 0:
            #     if no_of_layers % 4 == 0:
            #         # print("x shape: ", x.shape)
            #         x = layer(x, rope=False, ffn=True, mask=mask)
            #     x = layer(x, rope=True, ffn=True, mask=mask)
                
            #     # print("x shape: ", x.shape)
            # else:
            #     # print("x shape local: ", x.shape)
            #     if no_of_layers % 4 == 0:
            #         # print("x shape: ", x.shape)
            #         x = layer(x, rope=False, ffn=False, mask=mask)
            x, kv_cache = layer(x, kv_cache=kv_cache, ffn=None, mask=mask)
                # print("x shape local: ", x.shape)
            # no_of_layers += 1
        # print(x.shape)
        x = self.dropout(x)
        x = 2 * ((ModelArgs.no_of_decoder_layers) ** -0.5) * x
        x = self.norm(x)
        
        # if(inference):
        #     out = self.linear_layer(x)
        #     return out
        # if(ModelArgs.use_liger):  
        #     # print("yo")
        #     y = x.contiguous().view(-1, ModelArgs.embeddings_dims)
        #     if(actual_labels is not None):
        #         labels = actual_labels.contiguous().view(-1)
                
        #         # Pass linear layer weights FIRST as required [2][5]
        #         # ignore_index is already set during initialization
        #         loss = self.le_loss(self.linear_layer.weight, y, labels)
        #         return loss
        # else:
        #     # print("Hi")
        #     out = self.linear_layer(x)
        #     return out

        return x



class DeepSeekV3(nn.Module):
    def __init__(self,
                 device,
                 embeddings_dims: int = ModelArgs.embeddings_dims,
                 block_size: int = ModelArgs.block_size,
                 vocab_size: int = ModelArgs.vocab_size,
                 dropout = ModelArgs.dropout
                ):
        super().__init__()
        self.decoder = Block(device=device, embeddings_dims=embeddings_dims, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, block_size=block_size, vocab_size=vocab_size, dropout=dropout)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims, dtype=torch.float32, device=device)
        self.pos_embeddings = SinusoidalPositionalEmbeddings(embeddings_dims=embeddings_dims, device=device)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size, dtype=torch.float32, device=device, bias=False)
        # Weight tying - tie embedding and output projection weights
        self.embedding.weight = self.linear_layer.weight
        
        # Initialize the LigerFusedLinearCrossEntropyLoss for optimized training
        if ModelArgs.use_liger:
            # Initialize with ignore_index for padding tokens if enabled
            if ModelArgs.ignore_pad_token_in_loss:
                self.le_loss = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=tokenizer.pad_token_id
                )
            else:
                self.le_loss = LigerFusedLinearCrossEntropyLoss()

    def forward(self, x, inference=False, mask=None):
        if(mask is not None):
            x = x * mask
            
        x = self.embedding(x)
        x = x + self.pos_embeddings(x)  # Add positional embeddings
        B, T, C = x.shape
            
        if inference:
            # For inference, we only need the last token prediction
            decoder_out = self.decoder(x, mask=mask)
            logits = self.linear_layer(decoder_out)
            return logits
        else:
            decoder_out = self.decoder(x, mask=mask)
            logits = self.linear_layer(decoder_out)
            return logits
    


model = DeepSeekV3(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=ModelArgs.device)
model = model.to(ModelArgs.device)

# Log padding token handling
if ModelArgs.ignore_pad_token_in_loss:
    print(f"Ignoring padding token (ID: {tokenizer.pad_token_id}) in loss calculation")
else:
    print("Including padding tokens in loss calculation")

# model = DDP(model, device_ids=[gpu_ids])


#Printing a summary of the architecture
from torchinfo import summary
# from log_model_parameters import log_model_summary

idx, targets = get_batch('test')
idx = idx.to(ModelArgs.device)

# Print summary to console
print(summary(model=model,
        input_data=idx,
        # input_size=(ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.embeddings_dims),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]))

# Log summary to file
# log_model_summary(model, idx, "model_parameters.log")


def save_text(file_path, step, text):
    with open(file_path, 'w') as f:
        f.write(f"Step {step}: {text}\n")
        
        


save_checkpoint_iter = 2000
total_iters = 10000 * ModelArgs.epochs
eval_iters = 400
eval_check = 400
warmup_iters = 400 * ModelArgs.epochs
min_lr = 0.1 * ModelArgs.max_lr
lr_decay_iters = 10000 * ModelArgs.epochs  # Total iterations for learning rate decay
total_batch_size = 524288
micro_batch_size = ModelArgs.batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * (ModelArgs.block_size * 1))


torch.set_float32_matmul_precision('high')


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return ModelArgs.max_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (ModelArgs.max_lr - min_lr)


def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused


# import tqdm 
def train():
    # Initialize DDP if running with torchrun
    use_ddp = 'RANK' in os.environ
    if use_ddp:
        local_rank, world_size, rank, device = setup_ddp()
        print(f"Rank {rank}/{world_size} on device {device}")
    else:
        device = ModelArgs.device
        rank = 0
        world_size = 1
        print(f"Start running training on {device}.")
    
    # Initialize wandb for experiment tracking (only on rank 0)
    if rank == 0:
        wandb.init(
            project = 'DSV-Training',
            config = {
                'ignore_pad_token_in_loss': ModelArgs.ignore_pad_token_in_loss,
                'use_liger': ModelArgs.use_liger,
                'batch_size': ModelArgs.batch_size,
                'embeddings_dims': ModelArgs.embeddings_dims,
                'no_of_decoder_layers': ModelArgs.no_of_decoder_layers,
                'experts': ModelArgs.experts,
                'top_experts': ModelArgs.top_experts,
                'use_shared_expert': ModelArgs.use_shared_expert,
                'world_size': world_size
            }
        )
    
    # Create model and move to GPU
    model = DeepSeekV3(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=device)
    model = model.to(device)
    
    # Wrap model with DDP if using distributed training
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Get the underlying model for parameter groups (important for DDP)
    base_model = model.module if use_ddp else model
    
    hidden_weights = [p for p in base_model.decoder.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in base_model.decoder.parameters() if p.ndim < 2]
    nonhidden_params = [*base_model.linear_layer.parameters(), *base_model.embedding.parameters()]
    param_groups = [
        dict(params=hidden_weights, use_muon=True,
             lr=0.02, weight_decay=0.01),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
             lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim),
    ]
    optimizer = MuonWithAuxAdam(param_groups)

    if rank == 0:
        print("Model loaded")
    # Setup optimizer
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim, eps=ModelArgs.eps)
    
    # Training parameters
    # save_checkpoint_iter = 2000
    # total_iters = 610000
    # eval_iters = 1000
    model = torch.compile(model)
    
    # Training progress bar (only on rank 0)
    if rank == 0:
        train_epoch_iterator = tqdm.tqdm(range(total_iters), desc="Training")
    else:
        train_epoch_iterator = range(total_iters)
        
    val_dataloader = prepare_dataset('val', device, ModelArgs.batch_size, use_ddp=use_ddp)
    val_iterator = iter(val_dataloader)
    # Get batches for training
    @torch.inference_mode()
    def estimate_loss():
        out = {}
        model.eval()
        count = 0
        for split in ['val']:
            if rank == 0:
                print(f"Starting with {split} evaluation...")
            losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):

                nonlocal val_iterator
                
                # for k, batch in enumerate(dataloader):
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    batch = next(val_iterator)
            
                idx = batch["input_ids"].to(device)
                targets = batch["labels"].to(device)
                # mask = torch.ones(ModelArgs.batch_size, ModelArgs.block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the entire block
                # mask = mask.masked_fill(idx == tokenizer.pad_token_id, 0)  # Set padding tokens to 0 in the mask
                # if ModelArgs.use_liger:
                #     # Pass actual labels to the model to use optimized loss function
                #     # ignore_index is already set in the model's le_loss initialization
                #     loss = model(idx, actual_labels=targets)
                # else:
                # Standard cross entropy path
                # mask= torch.ones(ModelArgs.batch_size, ModelArgs.block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the entire block
                # mask = mask.masked_fill(idx == tokenizer.pad_token_id, 0)  # Set padding tokens to 0 in the mask
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Get the base model for DDP
                    eval_model = model.module if use_ddp else model
                    if ModelArgs.use_liger:
                        # Use Liger fused linear cross entropy loss
                        decoder_out = eval_model.decoder(eval_model.embedding(idx) + eval_model.pos_embeddings(eval_model.embedding(idx)), mask=None)
                        decoder_out_flat = decoder_out.contiguous().view(-1, ModelArgs.embeddings_dims)
                        targets_flat = targets.contiguous().view(-1)
                        # Make sure le_loss exists
                        if hasattr(eval_model, 'le_loss'):
                            loss = eval_model.le_loss(eval_model.linear_layer.weight, decoder_out_flat, targets_flat)
                        else:
                            # Fallback to standard cross entropy
                            logits = eval_model.linear_layer(decoder_out)
                            B, T, C = logits.shape
                            logits_flat = logits.contiguous().view(-1, C)
                            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=tokenizer.pad_token_id, reduction='mean')
                    else:
                        # Standard cross entropy loss
                        logits = eval_model(idx, mask=None)  # Get logits from the model [B, T, C]
                        B, T, C = logits.shape
                        
                        # Standard next-token prediction loss
                        logits_flat = logits.contiguous().view(-1, C)  # [B*T, C]
                        targets_flat = targets.contiguous().view(-1)   # [B*T]
                        
                        # Compute cross entropy loss
                        loss = F.cross_entropy(
                            logits_flat, 
                            targets_flat, 
                            ignore_index=tokenizer.pad_token_id,
                            reduction='mean'
                        )
                
                losses[k] = loss.item()
                # count += 1
            
            # Average losses across all processes if using DDP
            if use_ddp:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)
                
            out[split] = losses.mean()

        model.train()
        return out
    token_count = 0
    # Start training loop
    model.train()
    if rank == 0:
        print("Lessgoo...")
        print("gradient steps: ", gradient_accumulation_steps)
    dataloader = prepare_dataset('train', device, ModelArgs.batch_size, use_ddp=use_ddp)
    train_dataloader = iter(dataloader) 
    accumulated_loss = 0.0
    
    # if ModelArgs.use_compile:
    # model = torch.compile(model)
    if rank == 0:
        print("Model compiled")
    
    for epoch in range(ModelArgs.epochs):
        # Set epoch for DistributedSampler
        if use_ddp and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
        for step in train_epoch_iterator:
            # Periodically evaluate loss on train and val sets (only on rank 0)
            if ((step % eval_iters == 0 and step != 0) or step == total_iters - 1) and rank == 0:
                losses = estimate_loss()
                avg_val_loss = torch.Tensor([losses['val']]).to(device)
                print(f"step {step}: train loss {accumulated_loss:.4f}, val loss {losses['val']:.4f}")
                val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
                # Log metrics to wandb
                wandb.log({
                    "val_perplexity": val_perplexity,
                    # "val_step_loss": losses['train'],
                    "val_step_loss": losses['val'],
                    "step": step
                })
            
            # Save checkpoint periodically (only on rank 0)
            if step % save_checkpoint_iter == 0 and step != 0 and rank == 0:
                print(f"Saving the model checkpoint for step: {step}")
                # Save the base model state dict (unwrapped from DDP)
                save_model = model.module if use_ddp else model
                torch.save(save_model.state_dict(), f"checkpoint_{step}.pt")
                print("Checkpoint saved")
            
            # Initialize gradient accumulation
            accumulated_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            
            # Gradient accumulation loop
            for micro_step in range(gradient_accumulation_steps):
                # Get batch for training step
                try:
                    batch = next(train_dataloader)
                except StopIteration:
                    train_dataloader = iter(dataloader)
                    batch = next(train_dataloader)
                
                idx = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                
                token_count += idx.numel()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Get the base model for DDP
                    train_model = model.module if use_ddp else model
                    if ModelArgs.use_liger:
                        # Use Liger fused linear cross entropy loss
                        decoder_out = train_model.decoder(train_model.embedding(idx) + train_model.pos_embeddings(train_model.embedding(idx)), mask=None)
                        decoder_out_flat = decoder_out.contiguous().view(-1, ModelArgs.embeddings_dims)
                        targets_flat = targets.contiguous().view(-1)
                        # Make sure le_loss exists
                        if hasattr(train_model, 'le_loss'):
                            loss = train_model.le_loss(train_model.linear_layer.weight, decoder_out_flat, targets_flat)
                        else:
                            # Fallback to standard cross entropy
                            logits = train_model.linear_layer(decoder_out)
                            B, T, C = logits.shape
                            logits_flat = logits.contiguous().view(-1, C)
                            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=tokenizer.pad_token_id, reduction='mean')
                    else:
                        # Standard cross entropy loss
                        logits = train_model(idx, mask=None)  # Get logits from the model [B, T, C]
                        B, T, C = logits.shape
                        
                        # Standard next-token prediction loss
                        logits_flat = logits.contiguous().view(-1, C)  # [B*T, C]
                        targets_flat = targets.contiguous().view(-1)   # [B*T]
                        
                        # Compute cross entropy loss
                        loss = F.cross_entropy(
                            logits_flat, 
                            targets_flat, 
                            ignore_index=tokenizer.pad_token_id,
                            reduction='mean'
                        )
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                
                if micro_step % 10 == 0 and rank == 0:
                    print(f"Micro Batch: {micro_step}/{gradient_accumulation_steps}")
                    print(f"Step: {step}/{total_iters}")
                    print(f"Total tokens processed: {token_count}")
            
            # Synchronize accumulated loss across all processes if using DDP
            if use_ddp:
                loss_tensor = torch.tensor(accumulated_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                accumulated_loss = loss_tensor.item()
            
            # Update learning rate
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
            
            # Compute gradient norms before clipping
            grad_norm_value = 0.0
            if ModelArgs.clip != 0.0:
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )
                grad_norm_value = total_norm_before.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)
                
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )
                
                if step != 0 and rank == 0:
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")

            # Optimizer step after accumulating gradients
            optimizer.step()
            
            # Calculate perplexity from accumulated loss
            perplexity = torch.exp(torch.tensor(accumulated_loss)).item()
            
            # Log metrics to wandb (only on rank 0)
            if rank == 0:
                wandb.log({
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "Train_Loss": accumulated_loss,
                    "Train Perplexity": perplexity,
                    "Total Tokens Processed": token_count,
                    "Step": step,
                    "Gradient Norm": grad_norm_value,
                    "Gradient Accumulation Steps": gradient_accumulation_steps
                })
                
                if step % eval_iters == 0:
                    prompt = "Once upon a time there lived a baby deer named Bambi. "
                    generated_text = topk_sampling(train_model, prompt, max_length=ModelArgs.block_size, top_k=(50 * 2), temperature=0.9, device=device)
                    print(f" Step: {step} | Generated Text: {generated_text}")
                    save_text(f"generated_data/generated_text_{step}.txt", step, generated_text)
                    
        # Clean up DDP
        if use_ddp:
            cleanup_ddp()
            
        # Finish wandb run (only on rank 0)
        if rank == 0:
            wandb.finish()

# Print CUDA device count but won't be using DDP
world_size = torch.cuda.device_count()
print(f"CUDA devices available: {world_size}")

if __name__ == "__main__":
    train()