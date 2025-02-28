
# ORPO in Pytorch from scratch implementation

I Trained OPT-330M model using ORPO in Pytorch

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 2        | The number of samples processed before the model is updated.                |
| `max_lr`     | 8e-6     | Maximum learning rate.                                                      |
| `device`     | 'cuda:0' | The device to run the model on (e.g., 'cuda:0' for GPU).                    |
| `betas`      | 0.95,0.99| Beta values                                                                 |           
| `weight_decay`| 0.1     | Weight decay values for the optimizer                                       |


### Datasets

[UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)

### Frameworks:
**Pytorch**


### Epochs/Steps
Iterations (train) = nil

Val iterations = every 20


### Losses
Train loss - nil

Val loss - nil





