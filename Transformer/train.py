import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import torch.optim as optim
from transformers.models.albert import AlbertTokenizer

from datasets import load_dataset

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss



TOKEN = ''

def load_datasets(token, sample_size=None):
    """Load the datasets with optional sampling for faster analysis"""
    print("Loading datasets...")
    train_dataset = load_dataset("ai4bharat/samanantar", 'hi', split=f"train", token=token)

    print(f"Train dataset loaded")
    # print(f"Validation dataset loaded")
    
    return train_dataset

# Load datasets
fw_train = load_datasets(TOKEN)

fw_train = fw_train.train_test_split(test_size=0.01)

    # print(fw_train)
print(fw_train)
print(fw_train)


tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBARTSS")


#Hyperparameters

beta_2 = 0.98
eps = 1e-9
beta_1 = 0.9
block_size = 512
batch_size = 32
src_vocab_size = len(tokenizer)
tgt_vocab_size = len(tokenizer)
embeddings_dims = 512
# attn_dropout = 0.3  # Increased from 0.1 to reduce overfitting
no_of_heads = 8 #IMP needs to be thoroughly calculated
dropout = 0.1 # Increased from 0.1 to reduce overfitting
epochs = 1
max_lr = 6e-4  # Further reduced from 4e-4 to prevent instability
no_of_decoder_layers = 6 #IMP needs to be thoroughly calculated
attn_dropout = 0.1  # Increased dropout
weight_decay_optim = 0.1  # Slightly increased weight decay
device = 'cuda'
clip = 1.0  # Reduced from 1.0 for better stability
use_liger = True


def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        # "SCHEDULER_STATE": scheduler.state_dict(),  
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"snapshot_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")





def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        # max_length=ModelArgs.block_size,
        padding='longest',  # Changed to dynamic padding
        truncation=True,
        return_overflowing_tokens=True,
        return_tensors='pt'
    )




def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)
    # alpaca_prompt = '''
    

    def collate_fn(batch):
        # Extract text data
        # print(batch)
        en_texts = []
        hi_texts = []
        
        for item in batch:
            it =  tokenizer.bos_token + item['src'] + tokenizer.eos_token 
            en_texts.append(it)
            it = item['tgt'] + tokenizer.eos_token
            hi_texts.append(it)

        input_encodings = tokenizer(en_texts, padding='max_length', max_length=block_size, truncation=True,  add_special_tokens=False, return_tensors="pt")  # Let tokenizer add [BOS]/[EOS])
        target_encodings = tokenizer(hi_texts, padding='max_length', max_length=block_size, truncation=True,  return_tensors="pt", add_special_tokens=False)  # Let tokenizer add [BOS]/[EOS])
        input_encodings["labels"] = target_encodings["input_ids"].clone()  # Use `input_ids` as labels
        input_encodings["decoder_input_ids"] =  target_encodings["input_ids"].clone()
        # # input_encodings['decoder_input_ids'][:, 0] = tokenizer.bos_token_id 
        # # # input_encodings['decoder_input_ids'][:, -1] = tokenizer.eos_token_id
        # # input_encodings["decoder_input_ids"][:, 0] = tokenizer.bos_token_id  # Set the first token to BOS
        input_encodings["decoder_input_ids"][:, 1:] = input_encodings["decoder_input_ids"][:, :-1]  # Shift right
        input_encodings["decoder_input_ids"][:, 0] = tokenizer.bos_token_id  # Let the last token be end
        # # input_encodings['decoder_input_ids'][:, :-1] = input_encodings['decoder_input_ids'][:, 1:]  # Shift left
        
            # Fix label shifting
        # labels = target_encodings["input_ids"].clone()
        # decoder_input_ids = target_encodings["input_ids"].clone()
        
        # # Proper teacher forcing: shift labels left by 1
        # labels[:, :-1] = labels[:, 1:]
        # # labels[:, -1] = tokenizer.pad_token_id
        
        # input_encodings["labels"] = labels
        # input_encodings["decoder_input_ids"] = decoder_input_ids
        
        return input_encodings

        # input_encodings["labels"][:, -1] = tokenizer.eos_token_id  # Let the last token be end

        # return input_encodings


    dataloader = None

    if(split == 'train'):
        data_loader = DataLoader(
        fw_train['train'],
        batch_size=batch_size,
        
        
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
       
)
    elif(split == 'val'):
        data_loader = DataLoader(
        fw_train['test'],
        batch_size=batch_size,
            # generator=generator,
        # sampler=DistributedSampler(fw_train["test"]),
        collate_fn=collate_fn,
        # num_workers=os.cpu_count(),
        # num_workers = min(4, os.cpu_count()//2), # Don't overallocate
        # prefetch_factor = 2,  # Balance memory/performance
        drop_last=True,
        shuffle=False,
        # pin_memory=True,  # Add this
        # persistent_workers=True
    )
    return data_loader


# Sinusoidal Positional Embeddings
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        embeddings_dims=embeddings_dims,
        max_seq_len=block_size,
        theta=10000.0
    ):
        super().__init__()
        self.embeddings_dims = embeddings_dims
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Create the positional encoding matrix
        pe = torch.zeros(max_seq_len, embeddings_dims)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Calculate the div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, embeddings_dims, 2).float() * 
                           -(math.log(theta) / embeddings_dims))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embeddings_dims)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, embeddings_dims)
        seq_len = x.shape[1]
        # Access the registered buffer correctly
        pe = getattr(self, 'pe')
        return pe[:, :seq_len, :]


# Text embeddings
class TgtTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size = tgt_vocab_size,
        embeddings_dims = embeddings_dims
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings = tgt_vocab_size, embedding_dim=embeddings_dims, device=device) #Just a look up table to convert the toekns_ids to some numbers
        # nn.init.normal_(self.embeddings_table.weight.data, mean=0, std=0.02)

    def forward(self, x):
        return self.embeddings_table(x)
    

# Text embeddings
class SrcTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size = src_vocab_size,
        embeddings_dims = embeddings_dims
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings = src_vocab_size, embedding_dim=embeddings_dims, device=device) #Just a look up table to convert the toekns_ids to some numbers
        # nn.init.normal_(self.embeddings_table.weight.data, mean=0, std=0.02)

    def forward(self, x):
        return self.embeddings_table(x)
    


#Layer Normalization

class LayerNormalization(nn.Module):
    def __init__(
        self,
        embeddings_dims = embeddings_dims
    ):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=embeddings_dims)
    def forward(self, x):

        return self.norm(x)
    

#FeedForward Neural Network

class MLPBlock(nn.Module):
    def __init__(
        self,
        dropout = dropout,
        embeddings_size = embeddings_dims,
        # inner_dimensional_states: int = 3072
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(device=device, in_features=embeddings_size, out_features= 4 * embeddings_dims),
            nn.GELU(),
            nn.Linear(device=device, in_features= 4 * embeddings_dims, out_features=embeddings_size),
            nn.Dropout(p = dropout)
        )

    def forward(self, x):
        # mlp_weights_init = self.mlp.apply(weights_init)
        return self.mlp(x)
    
    

class MaskedAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)


    def forward(self, x, mask=None):
        batch, block_size, embd_dims = x.shape
    
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            
            masked_values = weights.masked_fill(mask == 0, float('-inf'))
            masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
            masked_values = masked_values.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            # weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            # out = self.dropout(out)
            return out
        else:
            masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
            weights = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            # weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            # out = self.dropout(out)
            return out
       


class MaskedMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.heads = nn.ModuleList([MaskedAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, x, mask=None):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out
    

#Single Attention Head

class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)


    def forward(self, k, v, q, srcmask=None):
        # Project inputs to queries, keys, values
        query = self.query(q)  # Query comes from decoder
        key = self.keys(k)     # Key comes from encoder
        value = self.values(v) # Value comes from encoder
        
        # Calculate attention scores
        weights = query @ torch.transpose(key, dim0=-2, dim1=-1) * (key.shape[-1] ** -0.5)

        if srcmask is not None:
            srcmask = srcmask.unsqueeze(1)
            masked_values = weights.masked_fill(srcmask == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            # weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ value
            # out = self.dropout(out)
            return out
        else:
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            # weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ value
            return out
        # out = self.dropout(out)
            # return out

        # # Apply mask if provided (for padding tokens)
        # if mask is not None:
        #     # Ensure mask has proper dimensions for broadcasting
        #     if mask.dim() == 2:
        #         mask = mask.unsqueeze(1)  # Add head dimension for broadcasting
        #     weights = weights.masked_fill(mask == 0, float('-inf'))
        
        # Unlike self-attention in decoder, cross-attention doesn't need causal masking
        # We want to attend to all encoder outputs
        
        # Apply softmax to get attention weights
        # weights_normalized = nn.functional.softmax(weights, dim=-1)
        # weights_normalized = self.dropout(weights_normalized)
        
        # Apply attention weights to values
        # out = weights_normalized @ value
        # return out
        
#Single Attention Head

class FullAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)


    def forward(self, x, mask=None):
        # batch, block_size, embd_dims = x.shape
        # print(x)
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        # masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        if(mask != None):
            mask = mask.unsqueeze(1)
            masked_values = weights.masked_fill(mask == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            # weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            # out = self.dropout(out)
            return out
        else:
            
            weights_normalized = nn.functional.softmax(weights, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            out = self.dropout(out)
        return out
        
        



class FullMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.heads = nn.ModuleList([FullAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, x, mask=None):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out
    



class CrossMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, value, key, x, srcmask=None):
        concat = torch.cat([head(value, key, x, srcmask) for head in self.heads], dim=-1)
        # print("Concat shape: ", concat.shape)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out
    

# Decoder Block

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
        dropout = dropout,
        # vocab_size = vocab_size
    ):
        super().__init__()

        self.cross = CrossMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.masked = MaskedMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        # self.layer_norm3 = LayerNormalization(embeddings_dims=embeddings_dims)
        self.layer_norm4 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, key, value, x, Srcmask=None, Targetmask=None):
        x = self.layer_norm1(x + self.masked(x, Targetmask)) #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = self.layer_norm2(x + self.cross(key, value, x, Srcmask)) #Very important step
        # x = x + self.mha(self.layer_norm1(x))  #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = self.layer_norm4(x + self.mlp_block(x)) #Very important step

        return x
    
    
# Decoder Block

class DecoderModel(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
        block_size = block_size,
        dropout = dropout,
        no_of_decoder_layers = no_of_decoder_layers,
        # vocab_size = vocab_size
    ):
        super().__init__()


        # self.positional_embeddings_tgt = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size


        # torch.nn.init.normal_(self.positional_embeddings_tgt, mean=0.0, std=0.02)

        # self.text_embds = TextEmbeddings(vocab_size=vocab_size, embeddings_dims=embeddings_dims)


        self.tgt_text_embds = TgtTextEmbeddings(vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims)
        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=tgt_vocab_size, device=device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        # self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        # Replace learned positional embeddings with sinusoidal embeddings
        self.positional_embeddings_tgt = SinusoidalPositionalEmbeddings(
            embeddings_dims=embeddings_dims, 
            max_seq_len=block_size, 
            theta=10000.0
        )
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer to prevent overfitting
        # out = self.decoder_layers(query, key, x)
        # Loop through each decoder layer
    def _init_weights(self, module):  #Weight Initialization
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, key, value, x, srcmask=None, target_mask=None):
        # print(mask.shape)
        # if(mask is not None):   
        #     x = x * mask
        # mask = mask.unsqueeze(-1) if mask is not None else None  # Add a dummy dimension for batch size
        x = self.tgt_text_embds(x)
        x= self.dropout(x)
        x = x + self.positional_embeddings_tgt(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(key, value, x, srcmask, target_mask)
        # x = self.layer_norm(x)
        x = self.dropout(x)
        return x
    



class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
        dropout = dropout,
        mask=None
    ):
        super().__init__()

        self.mha = FullMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, x, mask=None):
        x = self.layer_norm1(x + self.mha(x, mask))
        x = self.layer_norm2(x + self.mlp_block(x))

        return x
    
    


class EncoderModel(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
        block_size = block_size,
        dropout = dropout,
        no_of_decoder_layers = no_of_decoder_layers,
        # vocab_size = vocab_size
    ):
        super().__init__()

        # self.positional_embeddings_src = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size

        # torch.nn.init.normal_(self.positional_embeddings_src, mean=0.0, std=0.02)

        # self.text_embds = TextEmbeddings(vocab_size=vocab_size, embeddings_dims=embeddings_dims)

        # Replace learned positional embeddings with sinusoidal embeddings
        self.positional_embeddings_src = SinusoidalPositionalEmbeddings(
            embeddings_dims=embeddings_dims, 
            max_seq_len=block_size, 
            theta=10000.0
        )

        self.src_text_embeds = SrcTextEmbeddings(vocab_size=src_vocab_size, embeddings_dims=embeddings_dims)

        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer to prevent overfitting
    def _init_weights(self, module):  #Weight Initialization
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask):

        # print(x.shape)
        # if(mask is not None):   
        #     x = x * mask
        # mask = mask.unsqueeze(-1) if mask is not None else None  # Add a dummy dimension for batch size
        x = self.src_text_embeds(x)
        
        # print(self.positional_embeddings_src.shape)
        # print(x.shape)
        x = x + self.positional_embeddings_src(x)
        x = self.dropout(x)
        # print(x.shape)
        # Loop through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        x = self.dropout(x)
        return x




class Transformer(nn.Module):
    def __init__(
        self,

    ):
        super().__init__()

        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        self.norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=tgt_vocab_size, device=device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        if(use_liger):
          self.le_loss = LigerFusedLinearCrossEntropyLoss(
              ignore_index=tokenizer.pad_token_id
          )
        #  #weight tying
        # self.embeddings.weight = self.linear_layer.weight
    
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, src, tgt_idx, tgt, src_mask=None, tgt_mask=None, inference=False):
        # src_mask = src_mask.unsqueeze(-1) if src_mask is not None else None  # Add a dummy dimension for batch size
        # # tgt_mask = tgt_mask.unsqueeze(-1) if tgt_mask is not None else None  # Add a dummy dimension for batch size
        # if src_mask is not None:
        #     src = src * src_mask
        x = self.encoder(src, src_mask)
        # print("Encoder: ", x.shape)
        # if tgt_mask is not None:
        #     tgt_idx = tgt_idx * tgt_mask
            
        x = self.decoder(x, x, tgt_idx, src_mask, tgt_mask)
        # x = 2 * x * ((no_of_decox ** -0.5))
        x = self.norm(x)
        if(inference):
            out = self.linear_layer(x)
            return out
        if(use_liger):  
            # print("yo")
            y = x.contiguous().view(-1, embeddings_dims)
            if(tgt is not None):
                labels = tgt.contiguous().view(-1)
                
                loss = self.le_loss(self.linear_layer.weight, y, labels)
                return loss
        else:
            # Standard cross-entropy loss calculation
            batch_size, seq_len, _ = x.shape
            logits = self.linear_layer(x)
            
            # Reshape for cross entropy loss calculation
            logits = logits.view(-1, tgt_vocab_size)
            targets = tgt.contiguous().view(-1)
            
            # Calculate cross entropy loss with padding token ignored
            loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
            return loss






# from andrej karapathy github
def topk_sampling(model, prompt, device, max_length=30, top_k=50, temperature=0.8, repetition_penalty=1.2):
    model.eval()  # Set to eval mode for generation
    
    # prompt = tokenizer.bos_token + prompt + tokenizer.eos_token  # Add BOS and EOS tokens to the prompt
    # Tokenize the source text
    input_ids = tokenizer(prompt, add_special_tokens=True, max_length=max_length, padding='max_length', return_tensors="pt")['input_ids'].to(device)

    print("Input IDs: ", input_ids.shape)
    # Create an initial target sequence with just the BOS token
    target_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    src_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype).to(device)  # Create a mask of ones for the source sequence
    # tgt_mask = torch.ones(batch_size, block_size, dtype=input_ids.dtype).to(device)
    
    src_mask = src_mask.masked_fill(input_ids == tokenizer.pad_token_id, 0)
    # tgt_mask = tgt_mask.masked_fill(targets_idx == tokenizer.pad_token_id, 0)# Use the same mask for target sequence
                
    # src_mask = 
    # First, encode the source sequence - do this only once
    with torch.no_grad():
        encoder_output = model.encoder(input_ids, src_mask)
    
    # if max_length > len(input_ids[0]):
    #     max_length -= len(input_ids[0])   # Ensure at least one token is generated after the prompt
    # else:
    #     max_length = len(input_ids[0]) - max_length
        
    for i in range(max_length):
        with torch.no_grad():
            # Pass the encoder output and current target sequence to the decoder
            decoder_output = model.decoder(encoder_output, encoder_output, target_ids, src_mask, None)
            
            # Get logits from the final linear layer
            logits = model.linear_layer(model.norm(decoder_output))

            logits = logits[:, -1, :] / temperature  # Use only the last token's logits and apply temperature
            
            # Apply repetition penalty
            # if len(target_ids[0]) > 1:
            #     for token_id in set(target_ids[0].tolist()):
            #         logits[0, token_id] /= repetition_penalty
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            xcol = torch.gather(top_k_indices, -1, next_token)
            
            target_ids = torch.cat([target_ids, xcol], dim=1)
            
            # Check if we generated an EOS token
            if xcol.item() == tokenizer.eos_token_id:
                break
    print("Target IDs: ", target_ids.shape)
    model.train()  # Set back to train mode
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(target_ids[0])
    return generated_text


import copy

def beam_search_corrected(
    model,
    prompt: str,
    tokenizer,         # Pass tokenizer explicitly
    device: str,
    block_size: int,    # Pass block_size for source padding
    beam_width: int = 5,
    max_length: int = 50,
    temperature: float = 1.0
):
    model.eval()
    model = model.to(device) # Ensure model is on the correct device

    # 1. Tokenize and prepare source input with padding and mask
    # Add BOS/EOS to source consistent with your training collate_fn
    # src_text_with_special_tokens = tokenizer.bos_token + prompt + tokenizer.eos_token
    inputs = tokenizer(
        prompt,
        add_special_tokens=True,  # Add BOS and EOS tokens
        max_length=block_size,
        padding='max_length', # Pad to block_size
        truncation=True,
        return_tensors="pt"
    )
    src_input_ids = inputs["input_ids"].to(device)
    # This is the attention mask for the source sequence
    # src_attention_mask = inputs["attention_mask"].to(device)
    
    src_mask = torch.ones(src_input_ids.shape[0], src_input_ids.shape[1], dtype=torch.long, device=device)  # Create a mask of ones for the source sequence
    src_mask = src_mask.masked_fill(src_input_ids == tokenizer.pad_token_id, 0)  # Mask padding tokens

    with torch.no_grad():
        # Encoder output is computed once using the properly masked source
        encoder_output = model.encoder(src_input_ids, src_mask)
        # encoder_output shape: [1, block_size, embeddings_dims]

    # Initialize beams: list of (sequence_tensor, log_probability_score)
    # Sequence starts with BOS token
    initial_sequence = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    beams = [(initial_sequence, 0.0)]
    completed_sequences = []

    for step in range(max_length):
        if not beams: # Stop if no active beams left
            break

        # Store candidates for the next step from all current beams
        all_next_candidates = []

        for current_seq_tensor, current_score in beams:
            # If sequence already ended with EOS, move to completed and skip expansion
            if current_seq_tensor[0, -1].item() == tokenizer.eos_token_id:
                completed_sequences.append((current_seq_tensor, current_score))
                continue

            # Prepare decoder input (current target sequence)
            decoder_input_ids = current_seq_tensor # Shape: [1, current_target_length]

            # Target self-attention mask (causal is internal in your MaskedMHA if mask=None)
            # For active generation, decoder_input_ids won't have PADs.
            # An all-ones mask is fine here as causal masking is handled by torch.tril.
            # Or, (decoder_input_ids != tokenizer.pad_token_id).long()
            tgt_self_attention_mask = torch.ones_like(decoder_input_ids, device=device)


            with torch.no_grad():
                # encoder_output (key/value) has batch_size 1 and will broadcast
                # decoder_input_ids (x) has batch_size 1
                # The 'mask' to model.decoder will be used for target self-attention (causal handled internally)
                # and also for cross-attention in your current TransformerDecoderBlock.
                decoder_out = model.decoder(
                    encoder_output,
                    encoder_output,
                    decoder_input_ids,
                    src_mask # Or None if MaskedMHA handles None for causal perfectly
                )
                # Get logits for the *last* token in the sequence

                logits_last_token = model.linear_layer(model.norm(decoder_out[:, -1, :])) # Shape: [1, vocab_size]
                logits_last_token = logits_last_token / temperature
                log_probs = F.log_softmax(logits_last_token, dim=-1) # Shape: [1, vocab_size]

            # Get top k next tokens for *this* beam
            # Squeeze log_probs from [1, vocab_size] to [vocab_size] before topk
            top_k_log_probs, top_k_indices = torch.topk(log_probs.squeeze(0), beam_width, dim=-1)

            for i in range(beam_width): # For each of the top k choices for the current beam
                next_token_id = top_k_indices[i].item()
                next_token_log_prob = top_k_log_probs[i].item()

                new_seq_tensor = torch.cat(
                    [current_seq_tensor, torch.tensor([[next_token_id]], device=device)],
                    dim=1
                )
                new_score = current_score + next_token_log_prob
                all_next_candidates.append((new_seq_tensor, new_score))

        # Filter out completed sequences from the list of candidates for the next step
        # and update the list of active beams
        new_beams = []
        temp_completed = [] # Temporarily hold sequences completed in *this* step

        all_next_candidates.sort(key=lambda x: x[1], reverse=True) # Sort all potential next steps

        for cand_seq, cand_score in all_next_candidates:
            if cand_seq[0, -1].item() == tokenizer.eos_token_id:
                temp_completed.append((cand_seq, cand_score))
            elif len(new_beams) < beam_width: # Add to active beams if not EOS and we need more beams
                new_beams.append((cand_seq, cand_score))

            # Optimization: if we have enough active beams and completed ones,
            # we might not need to check low-score candidates.
            # if len(new_beams) == beam_width and len(temp_completed) >= beam_width:
            #     break
        beams = new_beams
        completed_sequences.extend(temp_completed)


        # Early stopping: if all active beams are worse than the k-th best completed sequence
        if len(completed_sequences) >= beam_width:
            completed_sequences.sort(key=lambda x: x[1], reverse=True)
            # If beams is empty or the best active beam is worse than the worst of top-k completed
            if not beams or (beams and completed_sequences[beam_width-1][1] > beams[0][1]):
                break

    # If no sequences were completed, use the active beams
    if not completed_sequences and beams:
        completed_sequences.extend(beams)
    elif not completed_sequences and not beams: # Should be rare
        model.train()
        return "[Beam Search Error: No sequences generated]"


    # Sort all completed sequences by score (descending)
    completed_sequences.sort(key=lambda x: x[1], reverse=True)

    if not completed_sequences: # Still possible if max_length is too short, etc.
        model.train()
        return "[Beam Search Error: No completed sequences found]"

    best_sequence_tensor = completed_sequences[0][0].squeeze(0) # from [1, seq_len] to [seq_len]

    model.train() # Set model back to training mode

    generated_text = tokenizer.decode(best_sequence_tensor)
    
    return generated_text

#Import the improved generation methods
# from improved_generation import generate_with_nucleus_sampling, beam_search_generation

#Instantiating the model
model = Transformer()
model = model.to(device)

# Printing a summary of the architecture
# !pip install torchinfo
from torchinfo import summary
# Create dummy inputs for all required arguments
src = torch.randint(
        low=0,
        high=src_vocab_size,
        size=(batch_size, block_size),
        dtype=torch.long
    ).to(device)

tgt = torch.randint(
        low=0,
        high=tgt_vocab_size,
        size=(batch_size, block_size),
        dtype=torch.long
    ).to(device)

# Create dummy masks (all ones for no masking in summary)
src_mask = torch.ones(batch_size, block_size, dtype=torch.long).to(device)  # Create a mask of ones for the source sequence
# summary(model=model,
#         input_data=(src, tgt, tgt),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])



# Clean up summary model
del model
torch.cuda.empty_cache()

# print("ghdgh")
def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused




def save_to_file(step, text):
    
    with open(f'data/generations_{step}.txt', 'w') as f:
        f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
        f.write(text + "\n\n")
        
    
#Train the  model





torch.set_float32_matmul_precision('high')

scaler = torch.amp.GradScaler(enabled=True)  # Enable mixed precision training

save_checkpoint_iter = 500
total_iters = 10000
eval_iters = 200
eval_check = 200
warmup_iters = 400  # Increased warmup for better stability
min_lr = 0.1 * max_lr
lr_decay_iters = 10000
total_batch_size = 524288  # Reduced to be more reasonable
micro_batch_size = batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * 1 * block_size)  # Fixed calculation

# Transformer learning rate scheduler from "Attention is All You Need" paper
# lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

# def get_lr(it):
#     # Avoid division by zero for step 0
#     step_num = max(it, 1)
    
#     # Transformer learning rate schedule
#     # d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
#     d_model_factor = embeddings_dims ** -0.5
#     step_factor = step_num ** -0.5
#     warmup_factor = step_num * (warmup_iters ** -1.5)
    
#     return d_model_factor * min(step_factor, warmup_factor)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)



def train():
   
#         # Initialise run
    wandb.init(
            # entity = 'rajceo2031',
                        project = 'Translation',
                        # config = CFG,
                        # save_code = True,
                        #group = 'ANN',
                        #job_type = 'train'
)
    print("wandb initialized")

    model = Transformer()
    # print(f"Model on device {device} is ready")
    print(f"Model on device {device} is ready")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(beta_1, beta_2),
        # weight_decay=weight_decay_optim,
        eps=eps,
        
    )
   
    model = torch.compile(model)

    model = model.to(device)

    

    
    
    model.eval()
    world_size = 1
    @torch.inference_mode()
    def estimate_loss(val_loader, val_iterator, device):
        out = {}
        # train_loader = prepare_dataset('train', ModelArgs.batch_size)
        
        # val_loader_iterator = iter(val_loader)
        loader = None
        epoch_loss = None
        epoch_losses = []
      
        # print("Starting the eval...")
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            # losses = torch.zeros(ModelArgs.val_epochs)
            # if(split == 'train'):
            #         loader = train_loader
            # if(split == 'val'):
            #         loader = val_loader
            for step in range(eval_check):  
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_loader_iterator = iter(val_loader)
                    batch = next(val_loader_iterator)
                
                total_loss = 0  
                # loader.sampler.set_epoch(step)
                total_batches = 0 
                # batch = next(val_loader_iterator)
                # for batch in loader:  # Loop through DataLoader batches
                idx = batch['input_ids'].to(device)
                targets_idx = batch['decoder_input_ids'].to(device)
                targets = batch['labels'].to(device)
                
                # print("Batch: ", idx)
                # print("Batch: ", targets_idx)
                src_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the source sequence
                tgt_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)
                
                src_mask = src_mask.masked_fill(idx == tokenizer.pad_token_id, 0)
                tgt_mask = tgt_mask.masked_fill(targets_idx == tokenizer.pad_token_id, 0)# Use the same mask for target sequence
                # token_count += idx.numel()
                with torch.autocast(device_type=device, dtype=torch.float16):

                    loss = model(idx, targets_idx, targets, src_mask, tgt_mask)
                total_loss += loss.item()
                total_batches += 1
                    
                    
                # print("Loss: ", loss.item())
            # Compute mean loss for this epoch
            epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
            epoch_losses.append(epoch_loss)

                # print(f"Epoch {epoch + 1}/{ModelArgs.val_epochs}: Loss = {epoch_loss:.4f}")

            # Compute mean loss across all evaluation epochs
            out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            epoch_loss = None
            epoch_losses = []

        model.train()
        return out

    # model = model.to(rank)
    model.train()
    count = 0
   
    train_dataloader = prepare_dataset('train', device, batch_size)
    val_loader= prepare_dataset('val', device, batch_size)
   
    print("Loaders ready both")


    
    # step = 0
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    token_count = 0
    tgt_mask = torch.ones(batch_size, block_size).to(device)  # Create a mask of ones for the target sequence
    src_mask = torch.ones(batch_size, block_size).to(device)  # Create a mask of ones for the source sequence
    
    # Calculate total batches per epoch and total batches across all epochs
    batches_per_epoch = len(train_dataloader)
    total_batches = batches_per_epoch * epochs
    
    # total_iters = len(train_dataloader)
    print(f"Training info: {epochs} epochs, {batches_per_epoch} batches per epoch, {total_batches} total batches")
    
    # Create overall progress bar
    overall_pbar = tqdm(total=total_batches, desc="Overall Training Progress", unit="batch")
    
    for epoch in range(epochs):
        # Create epoch progress bar
        # epoch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for step in tqdm(range(total_iters)):
            # every once in a while evaluate the loss on train and val sets
            if (step % eval_iters == 0) or step == total_iters - 1:
                losses = estimate_loss(val_loader, val_data_iterator, device)

                avg_val_loss = losses['val']

                print(f"[GPU {device}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")

                avg_val_loss = torch.Tensor([losses['val']]).to(device)

                all_gpus_avg_val_loss = avg_val_loss / world_size
                print(f"Val Loss: {all_gpus_avg_val_loss.item():.4f}")

                perplexity = torch.exp(torch.tensor(all_gpus_avg_val_loss.item()))  # Calculate perplexity

                wandb.log({
                        "All GPU Val_Loss": all_gpus_avg_val_loss.item(),
                        "Val Perplexity": perplexity.item(),
                        "Total Tokens Processed": token_count,
                        "Step": step,
                    })
                print(f"Step: {step} | All GPU Val Loss: {all_gpus_avg_val_loss.item():.4f} | Perplexity: {perplexity.item():.4f} | Tokens: {token_count}")

            if step % save_checkpoint_iter == 0:
                print(f"Saving the model checkpoint for step: {step}")
                _save_snapshot(model, optimizer, None, None, step)
            
            # Initialize gradient accumulation
            accumulated_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            
            # Gradient accumulation loop
            for micro_step in range(gradient_accumulation_steps):
                try:
                    # if micro_step == 0:
                    #     # Use the current batch for the first micro step
                    #     micro_batch = batch
                    # else:
                        # Get next batch for subsequent micro steps
                    micro_batch = next(train_data_iterator)
                except StopIteration:
                    # Reset iterator if we reach the end
                    train_data_iterator = iter(train_dataloader)
                    micro_batch = next(train_data_iterator)

                # Extract data from batch dictionary
                idx = micro_batch['input_ids'].to(device)
                targets_idx = micro_batch['decoder_input_ids'].to(device)
                targets = micro_batch['labels'].to(device)
                
                src_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)
                tgt_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)
                
                src_mask = src_mask.masked_fill(idx == tokenizer.pad_token_id, 0)
                tgt_mask = tgt_mask.masked_fill(targets_idx == tokenizer.pad_token_id, 0)
                
                token_count += idx.numel()
                
                with torch.autocast(device_type=device, dtype=torch.float16):
                
                # Forward pass
                    loss = model(idx, targets_idx, targets, src_mask, tgt_mask)

                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                accumulated_loss += loss.item()

                print(f" Mini Step: {micro_step} / {gradient_accumulation_steps} | Loss: {loss.item():.4f}")

                # Backward pass
                scaler.scale(loss).backward()

            # Check for unused parameters
            unused_params = find_unused_parameters(model)
            if unused_params:
                print(f"Unused parameters: {unused_params}")

            # Update learning rate
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
            
            # Compute gradient norms before clipping
            total_norm_before = 0.0
            if clip != 0.0:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

                # Compute gradient norms after clipping
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )
                
                if device == 0 and step != 0:
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")
            
            # Optimizer step (after accumulating gradients)
            # optimizer.step()
            scaler.step(optimizer)  # Use scaler to step optimizer
            scaler.update()  # Update the scaler after optimizer step
            
            # Increment step counter after gradient accumulation
            # step += 1
            
            torch.cuda.empty_cache()
        
            # Calculate metrics
            accumulated_loss /= world_size
            perplexity = torch.exp(torch.tensor(accumulated_loss))
            
            # if device == 0:
            print("Step : ", step, "/", total_iters)
            print("Total gradient accumulation steps: ", gradient_accumulation_steps)
            print("Total tokens processed: ", token_count)
            
            # Update progress bars
            # epoch_pbar.set_postfix({
            #     'Loss': f'{accumulated_loss:.4f}',
            #     'PPL': f'{perplexity.item():.2f}',
            #     'LR': f'{lr:.2e}',
            #     'Tokens': f'{token_count//1000}K'
            # })
            # overall_pbar.update(1)
            # overall_pbar.set_postfix({
            #     'Epoch': f'{epoch+1}/{epochs}',
            #     'Loss': f'{accumulated_loss:.4f}',
            #     'PPL': f'{perplexity.item():.2f}'
            # })
            
            # Log to wandb
            wandb.log({
                    "Learning Rate": lr,
                    "Train_Loss": accumulated_loss,
                    "Train Perplexity": perplexity.item(),
                    "Total Tokens Processed": token_count,
                    "Step": step,
                    "Gradient Norm": total_norm_before.item() if hasattr(total_norm_before, 'item') else total_norm_before,
                    "Gradient Accumulation Steps": gradient_accumulation_steps,
                })

            # Generate samples periodically
            if step != 0 and step % 500 == 0:
                count = 1
                while count:  
                    prompt = ["Hello! Myself an AI Assistant. How are you? ", "My name is Khan ", "How are you? ", "The AI will take our jobs ahhh! "]                   
                    # Use different generation methods and compare results
                    for prt in prompt:
                        print(f"\nGenerating text for prompt: {prt}")
                        generated_text = topk_sampling(model, prt, str(device), max_length=block_size, top_k=100, temperature=0.9, repetition_penalty=1.2)
                        beam_text = beam_search_corrected(model, prt, tokenizer, str(device), block_size=block_size, beam_width=5, max_length=block_size, temperature=0.8)

                        print(f"\nTop-K Generated Text: {generated_text}")
                        print(f"Beam Search Text: {beam_text}\n")
                        
                        with open(f'data/generations_{step}.txt', 'a') as f:
                            f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
                            f.write(f"Prompt: {prt}\n\n")
                            f.write(f"Top-K Generated Text: {generated_text}\n\n")
                            f.write(f"Beam Search Text: {beam_text}\n\n")
                    # generated_text = topk_sampling(model, prompt, str(device), max_length=block_size, top_k=100, temperature=0.8, repetition_penalty=1.2)
                    # beam_text = beam_search_corrected(model, prompt, tokenizer, str(device), block_size=block_size, beam_width=5, max_length=block_size, temperature=0.8)

                    # print(f"\nStep: {step} | Top-K Generated Text: {generated_text}")
                    # print(f"Step: {step} | Beam Search Text: {beam_text}\n")
                    
                    # with open(f'data/generations_{step}.txt', 'w') as f:
                    #     f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
                    #     f.write(f"Top-K Generated Text: {generated_text}\n\n")
                    #     f.write(f"Beam Search Text: {beam_text}\n\n")
                    
                    count -= 1
            # step += 1
            
        # Close epoch progress bar
        # epoch_pbar.close()

    # Close overall progress bar
    # overall_pbar.close()    
    _save_snapshot(model, optimizer, None, None, step)

    wandb.finish()
 
train()




