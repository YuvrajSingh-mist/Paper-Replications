# !pip install torchtune
# !pip install torchao
# !pip install wandb


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
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset


# Load model directly
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token='...')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#liger kernels
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


@dataclass
class ModelArgs:
    #Hyperparameters

    block_size = 512 
    batch_size = 32
    embeddings_dims = 512
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
    experts=16
    clip = 1.0
    top_experts=4
    noisy_topk = False
    use_checkpointing = False
    use_liger = True  # Use Liger kernels for optimized operations
    use_shared_expert = True  # Enable/disable shared expert
    ignore_pad_token_in_loss = True  # Whether to ignore padding tokens in loss calculation
    eps: float = 1e-8
    loss_scale = 0.3
    useauxFreeLoadBalancingLoss = True  
    aux_free_bias_update_rate = 0.001
    mtp_heads = 1  # Multi-token prediction heads
    latent_dim = 64  # Latent dimension for attention
#Datasets

# Using tinyshakespeare

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



#Subword level tokenization

# Loading custom trained BPE
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


# # create a mapping from characters to integers
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




def prepare_dataset(split, device, batch_size):
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
            data_loader = DataLoader(
            fw_train,
            # generator=generator,
            batch_size=batch_size,
             
            # sampler=DistributedSampler(fw_train, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False
        )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_test,
              
            
            batch_size=batch_size,
            # sampler=DistributedSampler(fw_test, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False
        )
    elif(fw):
        if(split == 'train'):
            data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            
            
            # sampler=DistributedSampler(fw_train['train'], shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True
    )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
                # generator=generator,
            # sampler=DistributedSampler(fw_train["test"]),
            collate_fn=collate_fn,
              
            drop_last=True,
            shuffle=True
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

        self.hidden_dims = ((embeddings_dims * 2) * 4 ) // 3  #Apply this when memory permits
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
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.ModuleList(DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers))
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        self.norm = Normalization(embeddings_dims)
        
        # Initialize the LigerFusedLinearCrossEntropyLoss for optimized training
        if ModelArgs.use_liger:
            # Initialize with ignore_index for padding tokens if enabled
            if ModelArgs.ignore_pad_token_in_loss:
                self.le_loss = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=tokenizer.pad_token_id
                )
            else:
                self.le_loss = LigerFusedLinearCrossEntropyLoss()

        #weight tying
        self.embeddings.weight = self.linear_layer.weight

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
        self.heads = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device) for _ in range(ModelArgs.mtp_heads)])
        self.dropout = nn.Dropout(p = dropout) 
        self.norm1 = nn.LayerNorm(embeddings_dims, eps=1e-6, device=device)
        self.norm2 = nn.LayerNorm(embeddings_dims, eps=1e-6, device=device)
        self.linear_layer = nn.Linear(in_features=2 * embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.unilayer = nn.ModuleList([DecoderLayer(device=device) for _ in range(ModelArgs.mtp_heads)])
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = Block(device=device, embeddings_dims=embeddings_dims, no_of_decoder_layers=ModelArgs.no_of_decoder_layers, block_size=block_size, vocab_size=vocab_size, dropout=dropout)
        self.pos_embeddings = SinusoidalPositionalEmbeddings(device=device, embeddings_dims=embeddings_dims, block_size=block_size, batch_size=ModelArgs.batch_size)
        # self.initialize_weights_embeddings()
        
    # def initialize_weights_embeddings(self):
        self.embedding.weight = self.decoder.embeddings.weight

    def forward(self, x, inference=False, mask=None):
        # B,T,C = x.shape
        mtp_heads_curr = 0
        if(mask is not None):
            x = x * mask
            # x = x.unsqueeze(-1)
            
        x = self.embedding(x)
        x = x + self.pos_embeddings(x)  # Add positional embeddings
        B, T, C = x.shape
        # if(mask is not None):
        #     mask = mask.unsqueeze(-1)
        #     x = x * mask  # Apply mask to the embeddings
            
        if inference:
            # For inference, we only need the last token prediction
            # Use the standard decoder path
            decoder_out = self.decoder(x)
            logits = self.decoder.linear_layer(decoder_out)
            return logits
        else:
            # For training, use multi-token prediction
            outputs = []
            
            for i in range(T - ModelArgs.mtp_heads):
                token_outputs = []
                for k in range(ModelArgs.mtp_heads):
                    # if i + k < T:
                    
                    if mtp_heads_curr == 0:
                        h_z = self.decoder(x[:, i + k + 1, :].unsqueeze(1), mask)  # Use decoder for the first head
                    else:
                        h_z = self.unilayer[k](x[:, i + k + 1, :].unsqueeze(1), mask)  # Add sequence dimension

                    # h_z = h_z.squeeze(1)  # Remove sequence dimension

                    # h_z = self.heads[k](x[:, i + k, :])
                    h_z = h_z.squeeze(1)
                    embed = x[:, i + k + 1, :]  # Add sequence dimension for the embedding
                    h_z = self.norm2(h_z)
                    embed = self.norm1(embed)
                    combined = torch.cat([embed, h_z], dim=-1)
                    # print("combined shape: ", combined.shape)
                    merged = self.linear_layer(combined)
                    # print("combined shape: ", combined.shape)
                    # merged = merged.squeeze(0)  # Remove sequence dimension umm its B,vocab size
                    logits = self.decoder.linear_layer(merged)  # Get logits from decoder
                    token_outputs.append(logits)
                
                if token_outputs: 
                    # Average the outputs from different heads for this position
                    avg_output = torch.stack(token_outputs, dim=1)
                    # print(avg_output.shape)
                    outputs.append(avg_output)
                
            # print("Output shape: ", len(outputs))
            # time.sleep(30)
            # if outputs:
            final_output = torch.stack(outputs, dim=0)
   
            final_output = final_output.permute(1, 0, 2, 3)  # Permute to [batch_size, T-D, seq_len, vocab_size]
            # print("shape: ", final_output.shape)
            # time.sleep(30)
            # print(fina[l_output.shape)
            # time.sleep(30)
            return final_output
            # else:
            #     # Fallback to standard decoder if no outputs
            #     decoder_out = self.decoder(x)
            #     return self.decoder.linear_layer(decoder_out)
    
    




# Instantiating the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# ModelArgs.device = device
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
# # from log_model_parameters import log_model_summary

idx, targets = get_batch('test')
idx = idx.to(ModelArgs.device)

# # Print summary to console
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
    # Set device to CUDA if available
    device = ModelArgs.device
    print(f"Start running training on {device}.")
    
    # Initialize wandb for experiment tracking
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
            'use_shared_expert': ModelArgs.use_shared_expert
        }
    )
    
    # Create model and move to GPU
    model = DeepSeekV3(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=ModelArgs.device)
    model = model.to(device)

    print("Model loaded")
    # Setup optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim, eps=ModelArgs.eps)
    
    # Training parameters
    # save_checkpoint_iter = 2000
    # total_iters = 610000
    # eval_iters = 1000
    # model = torch.compile(model)
    
    # Training progress bar
    train_epoch_iterator = tqdm.tqdm(range(total_iters), desc="Training")
    val_dataloader = prepare_dataset('val', device, ModelArgs.batch_size)
    val_iterator = iter(val_dataloader)
    # Get batches for training
    @torch.inference_mode()
    def estimate_loss():
        out = {}
        model.eval()
        count = 0
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            losses = torch.zeros(eval_iters)
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
                mask= torch.ones(ModelArgs.batch_size, ModelArgs.block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the entire block
                mask = mask.masked_fill(idx == tokenizer.pad_token_id, 0)  # Set padding tokens to 0 in the mask
                logits = model(idx, mask=None)  # Get logits from the model
                
                B, T, D, C = logits.shape
                
                # Vectorized loss calculation for multi-token prediction
                # Reshape logits to [B, T*D, C] using contiguous().view() to handle non-contiguous tensors
                logits_flat = logits.contiguous().view(B, T * D, C)
                
                # Create target indices for each position and head
                target_indices = []
                for i in range(T):
                    for k in range(D):
                        if i + k + 1 < targets.shape[1]:  # Ensure we don't go out of bounds
                            target_indices.append(i + k + 1)
                        else:
                            target_indices.append(targets.shape[1] - 1)  # Use last token if out of bounds
                
                # Convert to tensor and gather targets
                target_indices = torch.tensor(target_indices, device=targets.device)
                target_indices = target_indices.unsqueeze(0).expand(B, -1)  # [B, T*D]
                targets_flat = torch.gather(targets, 1, target_indices)  # [B, T*D]
                
                # Compute vectorized cross entropy loss
                loss = F.cross_entropy(
                    logits_flat.contiguous().view(-1, C), 
                    targets_flat.contiguous().view(-1), 
                    ignore_index=tokenizer.pad_token_id,
                    reduction='mean'
                )
                # loss *= ModelArgs.loss_scale  # Apply loss scaling if needed
                # batch_size, block_size, embeddings_dims = logits.shape
                
                # logits = logits.view(batch_size*block_size, embeddings_dims)
                
                # targets = targets.view(batch_size * block_size)
                
                # Use padding token as ignore_index if enabled
                # if ModelArgs.ignore_pad_token_in_loss:
                #     loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                # else:
                #     loss = nn.functional.cross_entropy(logits, targets)
                
                losses[k] = loss.item()
                # count += 1
            out[split] = losses.mean()

        model.train()
        return out
    token_count = 0
    # Start training loop
    model.train()
    print("Lessgoo...")
    print("gradient steps: ", gradient_accumulation_steps)
    dataloader = prepare_dataset('train', device, ModelArgs.batch_size)
    train_dataloader = iter(dataloader) 
    accumulated_loss = 0.0
    
    # if ModelArgs.use_compile:
    # model = torch.compile(model)
    print("Model compiled")
    
    for epoch in range(ModelArgs.epochs):
        for step in train_epoch_iterator:
            # Periodically evaluate loss on train and val sets
            if (step % eval_iters == 0 and step != 0) or step == total_iters - 1:
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
            
            # Save checkpoint periodically
            if step % save_checkpoint_iter == 0 and step != 0:
                print(f"Saving the model checkpoint for step: {step}")
                torch.save(model.state_dict(), f"checkpoint_{step}.pt")
                print("Checkpoint saved")
            
            # Get batch for training step
            try:
                batch = next(train_dataloader)
            except StopIteration:
                train_dataloader = iter(dataloader)
                batch = next(train_dataloader)
                
            # # for batch in dataloader:
            # input_ids = batch["input_ids"].to(device)
            # targets = batch["labels"].to(device)
            accumulated_loss = 0.0  
            # for micro_step in range(gradient_accumulation_steps):
            optimizer.zero_grad(set_to_none=True)  # Zero out gradients before backward pass
            try:
                batch = next(train_dataloader)
            except StopIteration:
                train_dataloader = iter(dataloader)
                batch = next(train_dataloader)
                
                
            idx = batch['input_ids'].to(device)
    
            targets = batch['labels'].to(device)
            
            token_count += idx.numel()
            
            mask = torch.ones(ModelArgs.batch_size, ModelArgs.block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the entire block\
            mask = mask.masked_fill(idx == tokenizer.pad_token_id, 0)  # Set padding tokens to 0 in the mask
            logits = model(idx, mask=None)
            
            B, T, D, C = logits.shape
            
            # Vectorized loss calculation for multi-token prediction
            # Reshape logits to [B, T*D, C] using contiguous().view() to handle non-contiguous tensors
            logits_flat = logits.contiguous().view(B, T * D, C)
            
            # Create target indices for each position and head
            # targets[:, i + k + 1] for all i in range(T) and k in range(D)
            target_indices = []
            for i in range(T):
                for k in range(D):
                    if i + k + 1 < targets.shape[1]:  # Ensure we don't go out of bounds
                        target_indices.append(i + k + 1)
                    else:
                        target_indices.append(targets.shape[1] - 1)  # Use last token if out of bounds
            
            # Convert to tensor and gather targets
            target_indices = torch.tensor(target_indices, device=targets.device)
            target_indices = target_indices.unsqueeze(0).expand(B, -1)  # [B, T*D]
            targets_flat = torch.gather(targets, 1, target_indices)  # [B, T*D]
            
            # Compute vectorized cross entropy loss
            loss = F.cross_entropy(
                logits_flat.contiguous().view(-1, C), 
                targets_flat.contiguous().view(-1), 
                ignore_index=tokenizer.pad_token_id,
                reduction='mean'
            )
            # loss *= ModelArgs.loss_scale  # Apply loss scaling if needed
            # logits = model(idx, inference=False)  # Get logits from the model
            
            # mask = torch.ones(ModelArgs.batch_size, ModelArgs.block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the entire block
            # mask = mask.masked_fill(idx == tokenizer.pad_token_id, 0)  # Set padding tokens to 0 in the mask
            # with torch.autocast(device_type=ModelArgs.device, dtype=torch.bfloat16):
            # Use LigerFusedLinearCrossEntropyLoss for efficient training
            # if ModelArgs.use_liger:
            #     # Pass actual labels to the model to use optimized loss function
            #     # ignore_index is already configured in model initialization
            #     loss = model(idx, actual_labels=targets)
            # else:
            #     # Standard cross entropy path
            #     logits = model(idx)
            #     batch_size, block_size, embeddings_dims = logits.shape
                
            #     logits = logits.view(batch_size*block_size, embeddings_dims)
                
            #     targets = targets.view(batch_size * block_size)
                
            #     # Use padding token as ignore_index if enabled
            #     if ModelArgs.ignore_pad_token_in_loss:
            #         loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
            #     else:
            #         loss = nn.functional.cross_entropy(logits, targets)
            
            # loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
            # accumulated_loss += loss.item()
            loss.backward()  
                
                
            # break
                # if(device == 0):
            #     if(micro_step % 10 == 0):
            # #     if(step == train_loader_length):
            # #       break
                    
            #         print("Micro Batch : ", micro_step)
            #         print("Step : ", step, "/", total_iters)
            #         print('Total batches: ', len(train_dataloader))
            #         print("Total gradient accumulation steps: ", gradient_accumulation_steps)
            #         print("Total tokens processed: ", token_count)
            #     # count += 1
            
            
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
            # optimizer.zero_grad(set_to_none=True)  # Zero out gradients before backward pass
            
            # unused_params = find_unused_parameters(model)
            # if unused_params:
            #         print(f"Unused parameters: {unused_params}")
            
            # Compute gradient norms before clipping
            if(ModelArgs.clip != 0.0):
                # scaler.unscale_(optimizer) #To avoid underflow
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)

                # Compute gradient norms after clipping
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )
                
                if(device  == 0 and step !=0):
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")

            optimizer.step()
            accumulated_loss += loss.item()  # Accumulate loss for logging
            # accumulated_loss = loss.item()
            # accumulated_loss /= gradient_accumulation_steps
            perplexity = torch.exp(torch.tensor(accumulated_loss)).item()  # Calculate perplexity
            # if(device == 0):
            wandb.log({
                        "Learning Rate": optimizer.param_groups[0]['lr'],
                        "Train_Loss": accumulated_loss,
                        # "Train loss": loss.item(),
                        "Train Perplexity": perplexity,
                        "Total Tokens Processed": token_count,
                        "Step": step,
                        "Gradient Norm": total_norm_before.item(),
                        # "Epoch": epoch
            })
            if(step % eval_iters == 0):
                    prompt = "Once upon a time there lived a baby deer named Bambi. "
                    generated_text = topk_sampling(model, prompt, max_length=ModelArgs.block_size, top_k=(50 * 2), temperature=0.9, device=device)


                    print(f" Step: {step} | Generated Text: {generated_text}")
                    save_text(f"generated_data/generated_text_{step}.txt", step, generated_text)
        # Finish wandb run
        wandb.finish()

# Print CUDA device count but won't be using DDP
world_size = torch.cuda.device_count()
print(f"CUDA devices available: {world_size}")
train()