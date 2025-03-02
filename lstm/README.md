
# LSTM in Pytorch from scratch implementation

Trained 128K LSTM model coded from scratch in Pytorch 

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 32       | The number of samples processed before the model is updated.                |
| `max_lr`     | 1e-4     | Maximum learning rate.                                                      |
| `dropout`    | 0.1      | Dropout.                                                                    |
| `epochs`     | 50       | Epochs                                                                      |           
| `block_size` | 64       | Sequence length                                                             |
| `No of neurons`     | 128       | Epochs                                                                      |   


### Frameworks:
**Pytorch**


### Epochs/Steps
Epochs (train) = 50

Val iterations = every epoch


### Losses

Train loss - 0.49 

Val loss - 0.48

### Loss Curves

![Train and Val loss curves](img/loss_curves.jpg)



