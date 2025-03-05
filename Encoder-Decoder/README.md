
# GRU in Pytorch from scratch implementation

Trained a GRU model coded from scratch in Pytorch 

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 16       | The number of samples processed before the model is updated.                |
| `max_lr`     | 1e-4     | Maximum learning rate.                                                      |
| `dropout`    | 0.2      | Dropout.                                                                    |
| `epochs`     | 50       | Epochs                                                                      |           
| `block_size` | 16      | Seq Len                                     |
| `No of neurons`| 16      | No of neurons in an GRU per layer                                          |    


### Frameworks:
**Pytorch**


### Epochs/Steps
Epochs (train) = 50

Val iterations = every epoch


### Losses

Train loss - 0.51 

Val loss - 0.48

<!-- ### Loss Curves

![Train and Val loss curves](img/loss_curves.jpg) -->



