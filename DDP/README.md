
# Trained Llama using DDP in Pytorch

I implemented a training loop and trained a Llama made from scratch using Data Distributed Parallel and torchrun.


### ModelArgs Hyperparameters

| Parameter              | Value         | Description                                                                 |
|------------------------|---------------|-----------------------------------------------------------------------------|
| `block_size`           | 128           | The size of each block.                                                     |
| `batch_size`           | 8             | The number of samples processed before the model is updated.                |
| `embeddings_dims`      | 768           | The dimensionality of the embeddings.                                       |
| `attn_dropout`         | 0.1           | Dropout rate for attention layers.                                          |
| `no_of_heads`          | 12            | Number of attention heads (needs thorough calculation).                     |
| `dropout`              | 0.1           | Dropout rate for the model.                                                 |
| `epochs`               | 100           | Number of training epochs.                                                  |
| `max_lr`               | 2.5e-4        | Maximum learning rate.                                                      |
| `no_of_decoder_layers` | 12            | Number of decoder layers (needs thorough calculation).                      |
| `weight_decay_optim`   | 0.1           | Weight decay for the optimizer.                                             |
| `beta_1`               | 0.9           | Exponential decay rate for the first moment estimates in the optimizer.     |
| `beta_2`               | 0.95          | Exponential decay rate for the second moment estimates in the optimizer.    |
| `clip`                 | 1.0           | Gradient clipping value.                                                    |
| `device`               | 'cuda'        | The device to run the model on (e.g., 'cuda' for GPU).                      |
| `no_kv_heads`          | 2             | Number of key-value heads.                                                  |
| `vocab_size`           | 10000         | Size of the vocabulary.                                                     |


### Datasets

**Tineshakespeare**: in the /data folder

### Frameworks:
**Pytorch**


### Epochs/Steps
Iterations (train) = 8000

Val iterations = every 100


### Losses
Train loss - 1.5

Val loss - 1.1

