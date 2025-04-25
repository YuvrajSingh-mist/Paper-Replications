
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import torch.optim as optim
import re
import os
from liger_kernel.transformers import LigerLayerNorm
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from transformers import AutoTokenizer
import jiwer 


HF_TOKEN = '...'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN)

tokenizer.add_special_tokens({"pad_token": "[PAD]"})


SOT = '<|startoftranscript|>'
EOT = '<|endoftranscript|>'
transcribe = '<|transcribe|>'
prev = '<|prev|>'

special_tokens_dict = {
    'additional_special_tokens': [SOT, EOT, transcribe, prev]
}


tokenizer.add_special_tokens(special_tokens_dict)

#Hyperparameters
epochs=2
block_size = 64
batch_size = 256
tgt_vocab_size = len(tokenizer)   
embeddings_dims = 512
attn_dropout = 0.1
no_of_heads = 4 
dropout = 0.1
max_lr = 1.5e-3
no_of_decoder_layers = 6 
attn_dropout = 0.1
weight_decay_optim = 0.1
log_mel_features = 80
kernel_size = 3
stride = (2,10)
sr = 16000
device= 'cuda:0'
SAMPLING_RATE=16000
N_MELS = 80  
WINDOW_DURATION = 0.025  # 25 milliseconds
STRIDE_DURATION = 0.010  # 10 milliseconds
max_t = 500
n_channels = N_MELS
clip = 1.0
use_flash_attention = True
use_liger = True
use_torch_compile = False 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
eps = 1e-6
beta_1 = 0.9
beta_2 = 0.98



torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


from jiwer import wer



from datasets import concatenate_datasets
from datasets import load_dataset


gs = load_dataset("speechcolab/gigaspeech", "s", token=HF_TOKEN, trust_remote_code=True)


print(gs)

# gs = gs['train'].train_test_split(shuffle=True,test_size=0.5)
validation_data = gs['validation']
combined = concatenate_datasets([gs['train'], gs['test']])

# Next, split the combined dataset so that the validation and test sets are about 1,000 each.
# We'll first split out 2,000 samples for validation+test.
split_result = combined.train_test_split(test_size=2000, shuffle=True, seed=42)
train_data = split_result['train']


# Now, split the 2,000 samples into two equal halves: 1,000 for validation and 1,000 for test.
split_temp = validation_data.train_test_split(test_size=0.5, shuffle=True, seed=42)
test_data = split_temp['test']


print(train_data)
print(validation_data)
print(test_data)

def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()





# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# tokenizer("hi")


MAX_DURATION_IN_SECONDS = 30

import librosa
from tqdm import tqdm
def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS

train_new_column = []

for x in tqdm(range(len(train_data))):
    train_new_column.append(librosa.get_duration(path=train_data[x]['audio']['path']))

gs_ = train_data.add_column("duration", train_new_column)


gs_ = gs_.filter(is_audio_length_in_range, input_columns=["duration"])


truncated_gs_train = gs_.remove_columns(["duration"])
# truncated_gs



val_new_column = []
# new_column = [librosa.get_duration(path=x) ]]
for x in tqdm(range(len(validation_data))):
    val_new_column.append(librosa.get_duration(path=validation_data[x]['audio']['path']))

gs_ = validation_data.add_column("duration", val_new_column)


gs_ = gs_.filter(is_audio_length_in_range, input_columns=["duration"])


truncated_gs_val = gs_.remove_columns(["duration"])
# truncated_gs

test_new_column = []
# new_column = [librosa.get_duration(path=x) ]]
for x in tqdm(range(len(test_data))):
    test_new_column.append(librosa.get_duration(path=test_data[x]['audio']['path']))

gs_ = test_data.add_column("duration", test_new_column)


gs_ = gs_.filter(is_audio_length_in_range, input_columns=["duration"])


truncated_gs_test = gs_.remove_columns(["duration"])


import numpy as np




def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        # "SCHEDULER_STATE": scheduler.state_dict(),  
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"snapshot_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")

def _load_snapshot(snapshot_path, model, optimizer, scheduler):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    # scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])  # Load scheduler state
    epoch = snapshot["EPOCHS_RUN"]
    step = snapshot["STEP_RUN"]
    print(f"Resuming from Epoch {epoch}, Step {step}")
    return epoch, step



def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        max_length=block_size,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )




def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)

    def collate_fn(batch):

        # MAX_FRAMES = int(MAX_DURATION_IN_SECONDS / STRIDE_DURATION)

        def pad_to_max_t(spectrogram, max_t):

            n_mels, t = spectrogram.shape
            if t < max_t:
                # Pad with zeros
                pad_width = ((0, 0), (0, max_t - t))
                spectrogram = np.pad(spectrogram, pad_width, mode='constant')
            else:
                spectrogram = spectrogram[:, :max_t]

            return spectrogram

        def clean(desc):
            # Use regex to remove anything between < and >
            cleaned_text = re.sub(r'<[^>]*>', '', desc)
            return cleaned_text

        # Audio processing parameters
        n_fft = int(SAMPLING_RATE * WINDOW_DURATION)
        hop_length = int(SAMPLING_RATE * STRIDE_DURATION)
        
        batch_spectrograms = []
        batch_input_ids = []
        batch_text = []
        batch_labels = []
        
        for item in batch:


            spectrogram = librosa.feature.melspectrogram(
                y=item['audio']['array'],
                sr=SAMPLING_RATE,
                n_mels=N_MELS,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                fmax=SAMPLING_RATE // 2
            )
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            SOT = '<|startoftranscript|>'
            EOT = '<|endoftranscript|>'
            transcribe = '<|transcribe|>'
            # prev = '<|prev|>'
            spectrogram = pad_to_max_t(spectrogram, block_size)
            # probs = round(random.random(),1)
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

            # if(probs == 0.5):
                # Normalize the spectrogram between -1 and 1
            spectrogram_min = spectrogram.min()
            spectrogram_max = spectrogram.max()
            # spectrogram = spectrogram.unsqueeze(0)  # Shape: (1, n_mels, max_t)
            # prev_text =
            text = clean(item['text'])
            original_text = text.lower()
            text = text.lower()
            text = SOT  + 'en' + transcribe +  text + EOT
            tokenized_text = tokenizer(text, truncation=True, padding='max_length', max_length=block_size, return_tensors='pt')
            # print(tokenized_text.shape)

            epsilon = 1e-8  # To avoid division by zero
            spectrogram = 2 * ((spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min + epsilon)) - 1

            # tokenized_win_prompt = tokenizer(text, max_length = ModelArgs.block_size, padding='max_length', truncation=True,  return_tensors="pt").to(device)
            tokenized_text['labels'] = tokenized_text['input_ids'].clone()
            tokenized_text['labels'][: , :-1] = tokenized_text['input_ids'][: , 1:]
            tokenized_text['labels'][: , -1] = tokenizer.eos_token_id

            tokenized_text_x = tokenized_text['input_ids'].squeeze(0)
            tokenized_text_y = tokenized_text['labels'].squeeze(0)

            batch_spectrograms.append(spectrogram)
            batch_input_ids.append(tokenized_text_x)
            batch_labels.append(tokenized_text_y)
            batch_text.append(original_text)
        return {
            "real_text": batch_text,
            'spectrogram': torch.stack(batch_spectrograms),
            'input_ids': torch.stack(batch_input_ids),
            'labels': torch.stack(batch_labels)
        }

    
    dataloader = None

    # if(tinystories):
    if(split == 'train'):
            data_loader = DataLoader(
            truncated_gs_train,
            # generator=generator,
            batch_size=batch_size,
             
            sampler=DistributedSampler(truncated_gs_train, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True
        )
    elif(split == 'val'):
            data_loader = DataLoader(
            truncated_gs_val,

            batch_size=batch_size,
            sampler=DistributedSampler(truncated_gs_val, shuffle=False),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True
        )

    elif(split == 'test'):
        data_loader = DataLoader(
            truncated_gs_test,
            batch_size=1,
            sampler=DistributedSampler( truncated_gs_test, shuffle=False),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True
        )
    
    return data_loader



#Position embeddings
class PositionEmbeddings(nn.Module):
    def __init__(
        self,
        embeddings_dims = embeddings_dims,
        block_size = block_size
    ):
        super().__init__()

        self.position_embeddings = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size
        # nn.init.normal_(self.position_embeddings.weight.data, mean=0, std=0.02)

    def forward(self, x):
        return self.position_embeddings




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




#Layer Normalization

class LayerNormalization(nn.Module):
    def __init__(
        self,
        embeddings_dims = embeddings_dims
    ):
        super().__init__()
        if(use_liger == False):
            self.norm = nn.LayerNorm(normalized_shape=embeddings_dims)
        else:
            self.norm = LigerLayerNorm(embeddings_dims)

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
        self.no_of_heads = no_of_heads
        if(use_flash_attention==False):
            self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
            self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=device, bias=False)
            self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)

        if(use_flash_attention):
            # Combined linear projections for Q, K, V
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=device)
            # self.out_proj = nn.Linear(embeddings_dims, embeddings_dims, bias=False, device=device)

    def forward(self, x):
        # print(x.shape)
        batch, block_size, embd_dims = x.shape
        if(use_flash_attention == False):
            k = self.keys(x)
            q = self.query(x)
            v = self.values(x)
        # if(use_flash_attention == False):
            masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout, is_causal=True
            )
            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch, block_size, -1)
            return out

            


class MaskedMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.heads = nn.ModuleList([MaskedAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=self.no_of_heads * embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, x):
        concat = torch.cat([head(x) for head in self.heads], dim=-1)
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
        self.no_of_heads = no_of_heads
        self.head_size = embeddings_dims // no_of_heads
        if(use_flash_attention == False):
            self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
            self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=device, bias=False)
            self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)

        # if(use_flash_attention):
        #     # Combined linear projections for Q, K, V
        #     self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=device)
            # self.out_proj = nn.Linear(embeddings_dims, embeddings_dims, bias=False, device=device)

    def forward(self, query, key, value, mask=None):


        batch, block_size, embd_dims = query.shape
        if(use_flash_attention == False):
            q = self.query(query)
            k = self.keys(key)
            v = self.values(value)


        if(use_flash_attention):

            batch, q_seq_len, _ = query.shape
            _, k_seq_len, _ = key.shape
            _, v_seq_len, _ = value.shape
            q = query.view(batch, q_seq_len, self.no_of_heads, self.head_size).transpose(1, 2)
            k = key.view(batch, k_seq_len, self.no_of_heads, self.head_size).transpose(1, 2)
            v = value.view(batch, v_seq_len, self.no_of_heads, self.head_size).transpose(1, 2)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout, is_causal=False
            )
            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch, q_seq_len, -1)
            return out
        else:   
            masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out


        #Single Attention Head

class FullAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.head_size = embeddings_dims // no_of_heads
        if(use_flash_attention == False):
            self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
            self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=device, bias=False)
            self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device,bias=False)
        self.dropout = nn.Dropout(p = attn_dropout)
        if(use_flash_attention):
            # Combined linear projections for Q, K, V
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=device)
            # self.out_proj = nn.Linear(embeddings_dims, embeddings_dims, bias=False, device=device)


    def forward(self, x, mask=None):
        batch, block_size, embd_dims = x.shape
        if(use_flash_attention == False):
            k = self.keys(x)
            q = self.query(x)
            v = self.values(x)
        # masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
        if(use_flash_attention):
  
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(batch, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout, is_causal=False
            )
            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch, block_size, -1)
            return out

        else:
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            if(mask != None):
                mask = mask.unsqueeze(1)
                masked_values = weights.masked_fill(mask == 0, float('-inf'))
                weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
                # weights_normalized = self.dropout(weights_normalized)
                out = weights_normalized @ v
                out = self.dropout(out)
                return out
            else:
                weights_normalized = nn.functional.softmax(weights, dim=-1) #Normalize along the embeddings dimension for all the tokens
                # weights_normalized = self.dropout(weights_normalized)
                out = weights_normalized @ v
                out = self.dropout(out)
                return out


            
class   FullMHA(nn.Module):
    def __init__(
        self,
        attn_dropout = attn_dropout,
        embeddings_dims = embeddings_dims,
        no_of_heads = no_of_heads,
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.heads = nn.ModuleList([FullAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features= self.no_of_heads * embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

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
        self.no_of_heads = no_of_heads
        self.heads = nn.ModuleList([CrossAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=self.no_of_heads * embeddings_dims, out_features=embeddings_dims, device=device, bias=False)

    def forward(self, value, key, x, mask=None):
        concat = torch.cat([head(x, key, value,  mask) for head in self.heads], dim=-1)
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

    def forward(self, key, value, x, mask=None):
        masked_out = self.masked(x)
        cross_out = self.cross(value, key, x, mask)
        x = x + self.layer_norm1(masked_out) #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        # print(x.shape)
        x = x + self.layer_norm2(cross_out) #Very important step
        # print(x.shape)
        # x = x + self.mha(self.layer_norm1(x))  #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = x + self.layer_norm4(self.mlp_block(x)) #Very important step
        # print(x.shape)

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




        # self.tgt_text_embds = TgtTextEmbeddings(vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims)
        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=tgt_vocab_size, device=device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        # self.positional_embeddings_tgt = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size
        self.positional_embeddings_tgt = PositionEmbeddings()
        # torch.nn.init.normal_(self.positional_embeddings_tgt, mean=0.0, std=0.02)

        # out = self.decoder_layers(query, key, x)
        # Loop through each decoder layer
    def _init_weights(self, module):  #Weight Initialization
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, key, value, x, mask):
        # x = self.tgt_text_embds(x)
        x = x + self.positional_embeddings_tgt(x)[:, :x.shape[1], :]
        # print(x.shape)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(key, value, x, mask)
        x = self.layer_norm(x)

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
        # print(self.mha(x, mask).shape)
        # print(x.shape)
        mha_out = self.mha(x, mask)
        # mha_out = mha_out
        # print(mha_out.shape)
        x = x + self.layer_norm1(mha_out)
        x = x + self.layer_norm2(self.mlp_block(x))

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

        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=embeddings_dims, kernel_size=kernel_size, device=device, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embeddings_dims, out_channels=embeddings_dims, kernel_size=kernel_size, device=device, padding=1)

        self.positional_embeddings_src = PositionEmbeddings()

        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)

    def _init_weights(self, module):  #Weight Initialization
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask):

        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)
        # print("Shape: ", x.shape)
        # x = self.src_text_embeds(x)
        # print(self.positional_embeddings_src.shape)
        x = x.permute(0, 2, 1)
        # print("Shape: ", x.shape)
        # print(self.positional_embeddings_src(x).shape)
        x = x + self.positional_embeddings_src(x)
        # print(x)
        # print(x.shape)
        # Loop through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x



        

class Whisper(nn.Module):
    def __init__(
        self,

    ):
        super().__init__()

        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        # self.pos = PositionalEmbeddings()
        self.tgt_text_embds = TgtTextEmbeddings(vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=tgt_vocab_size, device=device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        self.le_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id
        ).to(device)
        # self.src_text_embeds = SrcTextEmbeddings(vocab_size=src_vocab_size, embeddings_dims=embeddings_dims)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, actual_labels=None, inference=False):
        # print("Here: ", src.shape)
        # print("Here2: ", tgt.shape)
        # x = self.src_text_embeds(src)
        x = self.encoder(src, src_mask)

        y = self.tgt_text_embds(tgt)
        # print(x.shape)
        y = self.decoder(x, x, y, tgt_mask)
        # print(y.shape)
        if(inference):
            out = self.linear_layer(y)
            return out
        if(use_liger):  
            y = y.contiguous().view(-1, embeddings_dims)
            labels = actual_labels.contiguous().view(-1)
            
            # Pass linear layer weights FIRST as required [2][5]
            loss = self.le_loss(self.linear_layer.weight, y, labels)
            return loss
        else:
            out = self.linear_layer(y)
            return out


# test_iter = iter(test_loader)
def topk_sampling(model, batch, max_length=300, top_k=50, temperature=1.0, device='cuda'):
    # Get test batch (batch_size=1)
   


    # Extract inputs
    spectrogram = batch['spectrogram'].to(device)  # [1, n_mels, time]
    input_text = "<|startoftranscript|>en<|transcribe|>"  # Initial prompt
    global block_size
    # Tokenize initial input
    input_ids = tokenizer(
        input_text,
        return_tensors='pt',
        # max_length=block_size,
        # truncation=True,
        # padding='max_length'
    ).input_ids.to(device)
    len_input_id = len(input_ids)
    model.eval()
    generated_ids = input_ids.clone()
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Generation loop
        # print("Now: ", len(generated_ids))
        for _ in range(max_length):
            # Forward pass through full model
            outputs = model(
                src=spectrogram,  # Spectrogram input
                tgt=generated_ids,  # Text tokens
                src_mask=None,
                tgt_mask=None,
                actual_labels=None,
                inference=True
            )
            
            # Get last token logits
            logits = outputs[:, -1, :]
            
            # Apply temperature scaling
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            # Top-k filtering
            probs, indices = torch.topk(probs, top_k)
            # indices = indices
            
            
            # Sample from top-k
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, next_token)
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            # print("Generated ids: ", generated_ids.shape)
            # print(generated_ids)
            # if(generated_ids.shape[1] >= block_size):
            #     break
            # Stop if EOT generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode and clean
    transcript = tokenizer.decode(
        generated_ids[0], 
        skip_special_tokens=True
    )
    real_sentence = batch['real_text'][0]
    
    return real_sentence, transcript

def beam_search(model, tokenizer, prompt, beam_width=5, max_length=50, temperature=1.0):
    device = next(model.module.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    beam_scores = torch.zeros(beam_width, device=device)
    beam_sequences = input_ids.repeat(beam_width, 1)

    for _ in range(max_length):
        outputs = model(beam_sequences)
        logits = outputs[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)

        # Expand beams
        beam_scores = beam_scores.unsqueeze(-1) + torch.log(top_probs)
        beam_scores = beam_scores.view(-1)
        top_indices = top_indices.view(-1)

        # Select top beams
        beam_scores, top_beams = torch.topk(beam_scores, beam_width)
        beam_sequences = torch.cat([beam_sequences[top_beams // beam_width], top_indices[top_beams].unsqueeze(-1)], dim=-1)

    # Return the best sequence
    best_sequence = beam_sequences[0]
    return tokenizer.decode(best_sequence, skip_special_tokens=True)


model = Whisper()
model = model.to(device)

# Printing a summary of the architecture

# from torchinfo import summary

# idx = torch.randint(
#         low=0,
#         high=vocab_size,
#         size=(batch_size, block_size),
#         dtype=torch.long
#     )

# idx = idx.to(device)

# summary(model=model,
#         input_data=idx,

#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])


def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused

def greedy_decode(
    model, 
    tokenizer, 
    prompt, 
    device,
    max_length=50, 
    repetition_penalty=1.2, 
    context_window=10, 
    temperature=1.0, 
    eos_token_id=None,
    
):
  
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    generated_tokens = []
    eos_token_id = eos_token_id or tokenizer.eos_token_id  # Use EOS token if provided
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model.module(input_ids)
            logits = outputs[:, -1, :]  # Get logits for the last token

         
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated_tokens.append(next_token.item())

            # Stop if EOS token is generated
            # if next_token.item() == eos_token_id:
            #     break

            # Append the new token to the input
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the generated tokens
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)



def save_to_file(text):
    
    with open('generations.txt', 'a') as f:
        f.writelines(text + "\n\n")
        
    
#Train the  model


# writer = SummaryWriter(log_dir="runs/experiment")

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# Warmup phase for 2000 steps
def warmup_fn(step):
    if step < 2000:
        return step / 2000  # LR gradually increases
    return 1.0




torch.set_float32_matmul_precision('high')

scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

save_chechpoint_iter = 50
total_iters = 20000
eval_iters = 50
eval_check = 50
warmup_iters = 700
min_lr = 3e-6
lr_decay_iters = 20000
total_batch_size = 32768
micro_batch_size = batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * (block_size * torch.cuda.device_count()))

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

model.eval()
world_size = torch.cuda.device_count()


def train():
    setup()
    device = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(int(device))
    # torch.set_default_device('cuda')
    # train_dataloader = prepare_dataset(batch_size)
    # rank = torch.distributed.get_rank()
    print(f"Start running DDP on rank {device}.")
    # # create model and move it to GPU with id rank
    # device_id = rank % torch.cuda.device_count()
    # CFG = ModelArgs()

    global batch_size
    
    if(device == 0):

       
    
#         # Initialise run
        wandb.init(
            # entity = 'rajceo2031',
                        project = 'Whisper-DDP',
                        # config = CFG,
                        # save_code = True,
                        #group = 'ANN',
                        #job_type = 'train'
)
    print("wand initialized")
    
    model = Whisper()
    
    # print(f"Model on device {device} is ready")
    print(f"Model on device {device} is ready")

    optimizer = optim.AdamW(model.parameters(), lr=max_lr, betas=(beta_1, beta_2), weight_decay=weight_decay_optim, eps=eps, fused=True)
    
    if(use_torch_compile):
        model = torch.compile(model)
    
    model = model.to(device)
    
    model = DDP(model, device_ids=[device])
    

  
    def compute_wer(reference, hypothesis):

        error = jiwer.wer(reference, hypothesis)
        return error 
    
    model.eval()
    world_size = torch.cuda.device_count()
    @torch.inference_mode()
    def estimate_loss(val_loader, val_iterator, device):
        out = {}
        # train_loader = prepare_dataset('train', batch_size)
        
        # val_loader_iterator = iter(val_loader)
        loader = None
        epoch_loss = None
        epoch_losses = []
        # print("Starting the eval...")
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            # losses = torch.zeros(val_epochs)
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
                targets = batch['labels'].to(device)
                spec = batch['spectrogram'].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    
                    loss = model(spec, idx, actual_labels=targets)
                    # batch_size, block_size, embeddings_dims = logits.shape
                    # logits = logits.view(batch_size * block_size, embeddings_dims)  # Flatten tokens
                    # targets = targets.view(batch_size * block_size)

                    # loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)

                    total_loss += loss.item()
                    total_batches += 1

            # Compute mean loss for this epoch
            epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
            epoch_losses.append(epoch_loss)

                # print(f"Epoch {epoch + 1}/{val_epochs}: Loss = {epoch_loss:.4f}")

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
    test_loader = prepare_dataset('test', device, None)

    # for step in tqdm(range(total_iters)):
    # for epoch in range(epochs):
        # torch.cuda.synchronize() 
    
    # train_dataloader.sampler.set_epoch(epoch)
    
    # val_loader.sampler.set_epoch(epoch)
    print("Loaders ready both")
    # epochs = epochs

    train_loader_length = 0
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    test_iter = iter(test_loader)
    token_count = 0
    if(device == 0):
        train_loader_length = len(train_dataloader)
    
    for epoch in range(epochs):
        for step in tqdm(range(total_iters)):
        # print("Dataloader things: ", batch)
        # print("Total batches: ", len(train_dataloader))
            
            
            if(device == 0):
                # if(step % 100 == 0):
            #     if(step == train_loader_length):
            #       break
                    print("Step : ", step, "/", total_iters)
                    print('Total batches: ', len(train_dataloader))
                    print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                    print("Total tokens processed: ", token_count)
                    
            # all_gpus_avg_train_loss = None
            # all_gpus_avg_val_loss = None
            # every once in a while evaluate the loss on train and val sets
            if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
                losses = estimate_loss( val_loader, val_data_iterator, 'cuda')
                # avg_train_loss = losses['train']
                avg_val_loss = losses['val']
                # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # if device == 0:  # Only print on main process
                print(f"[GPU {device}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")
                # print(f"[GPU {device}] | Epoch {epoch}/{epochs}| |Step: {step} | Train Loss: {losses['train']:.4f}")
                    # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    # Log training loss more frequently
                    # Aggregate average loss across all G   PUs
                # avg_train_loss = torch.Tensor([losses['train']]).to(device)
                avg_val_loss = torch.Tensor([losses['val']]).to(device)
                # torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                
                if device == 0:
                    # all_gpus_avg_train_loss = avg_train_loss / world_size
                    # print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
                    all_gpus_avg_val_loss = avg_val_loss / world_size
                    print(f"All_GPUs_Val_losses: {all_gpus_avg_val_loss.item():.4f}")
                    
                
                    wandb.log({
                        # "Learning Rate": optimizer.param_groups[0]['lr'],
                        # "All_GPUs_Train_losses": all_gpus_avg_train_loss,
                        "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                        # "training_step_loss": losses['train'],
                        "val_step_loss": losses['val'],
                        # "Step": step,
                        # "Epoch": epoch
                    })
                
                
            
        

            if step % save_chechpoint_iter == 0 and device == 0 and step != 0:
                print(f"Saving the model checkpoint for step: {step}")
                _save_snapshot(model, optimizer, None, None, step)
            
            accumulated_loss = 0.0
            
            
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(gradient_accumulation_steps):
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    batch = next(train_data_iterator)
                # print(batch)
                # batch = next(train_data_iterator)
                # print(batch)
                # batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                idx = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                spec = batch['spectrogram'].to(device)
                token_count += len(idx)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = model(spec, idx, actual_labels=targets)
                    # batch_size, block_size, embeddings_dims = logits.shape
                    # print(logits.shape)
                    # print(targets)
                    # logits = logits.view(batch_size*block_size, embeddings_dims)
                    # print("OK")
                    # targets = targets.view(batch_size * block_size)
                    # print("OK2")
                    # loss = nn.functional.cross_entropy(loss, targets, ignore_index=tokenizer.pad_token_id)
                    
                    loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
                    accumulated_loss += loss.detach()
                
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # so that we dont synchronize the gradient everytime across the GPU devices
                scaler.scale(loss).backward()
                    # Check for unused parameters
                unused_params = find_unused_parameters(model)
                if unused_params:
                    print(f"Unused parameters: {unused_params}")
            # break
        
                if(device == 0):
                    if(micro_step % 10 == 0):
                #     if(step == train_loader_length):
                #       break
                        
                        print("Micro Batch : ", micro_step)
                        print("Step : ", step, "/", total_iters)
                        print('Total batches: ', len(train_dataloader))
                        print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                        print("Total tokens processed: ", token_count)
                # count += 1
        
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
                
            
            
            # Compute gradient norms before clipping
            if(clip != 0.0):
                scaler.unscale_(optimizer) #To avoid underflow
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

                # Compute gradient norms after clipping
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
                )
                
                if(device  == 0 and step !=0):
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")

            scaler.step(optimizer)
            scaler.update()
        
            # optimizer.step()
            # new_scheduler.step()
            # reference = batch['text']
            # hypothesis = logits
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            

            if(device == 0):
                wandb.log({
                        "Learning Rate": lr,
                        "All_GPUs_Train_losses": accumulated_loss.item(),
                        # "All_GPUs_Val_losses": all_gpus_avg_val_loss,
                        # "training_step_loss": losses['train'],
                        # "val_step_loss": losses['val'],
                        # "WER": wer,
                        "Step": step,
                        "Grad Norm": total_norm_before.item(),
                        # "Epoch": epoch
                        
                    })
            
            if device == 0 and step % 20 == 0:
                count = 2
                try:
                    batch = next(test_iter)
                except StopIteration:
                    test_loader_iterator = iter(test_loader)
                    batch = next(test_loader_iterator)
                        
                while(count):  
                    # prompt = "Once upon a time"
                    reference, generated_text = topk_sampling(model, batch, max_length=50, top_k=50, temperature=1.0, device=device)
                    wer = compute_wer(reference, generated_text)
                    # print(f" Step: {step} | Generated Text: {generated_text}")
                    print(f" Step: {step} | WER: {wer}")
                    wandb.log({
                        "Val WER": wer
                    })
        
                    print(f" Step: {step} | Generated Text: {generated_text} | Real Text: {reference}")

            
                    count -= 1
            
     
    if device == 0:
        wandb.finish()
    cleanup()


world_size = torch.cuda.device_count()
print(f"World size: {world_size}")
train()




