
# Trained Qwen0.5B-Instruct using Direct Preference Optimization in Pytorch

I implemented a training loop and trained a Llama made from scratch using Data Distributed Parallel and torchrun.


##  ModelArgs Hyperparameters

| Parameter              | Value         | Description                                                                 |
|------------------------|---------------|-----------------------------------------------------------------------------|

| `batch_size`           | 2            | The number of samples processed before the model is updated.                |
| `max_lr`               | 1e-6          | Maximum learning rate.                                                      |
| `device`               | 'cuda:0'      | The device to run the model on (e.g., 'cuda:0' for GPU).                    |

### Datasets

[UltraFeedback](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)

### Frameworks:
**Pytorch**


### Epochs/Steps
Iterations (train) = 3000

Val iterations = every 20


### Losses
Train loss - 0.67

Val loss - 0.68


### Local setup



