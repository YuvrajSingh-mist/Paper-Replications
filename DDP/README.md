
# Trained Llama using DDP in Pytorch

I implemented a training loop and trained a Llama made from scratch using DPP and torchrun.

### Model config

block_size = 128
batch_size = 8
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
clip = 1.0
device = 'cuda'
no_kv_heads = 2
vocab_size = 10000



### Datasets

**Tineshakespeare**: in the /data folder

### Frameworks:
**Pytorch**

Epochs = 5

Train loss
Val loss 


