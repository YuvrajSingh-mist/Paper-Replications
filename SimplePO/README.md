
# SimplePO in Pytorch from scratch implementation

Trained OPT-330M model using SimplePO in Pytorch for Instruction Following

[SimplePO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

## ModelArgs Hyperparameters

| Parameter    | Value    | Description                                                                 
|--------------|----------|-----------------------------------------------------------------------------|
| `batch_size` | 128        | The number of samples processed before the model is updated.                |
| `max_lr`     | 2e-5     | Maximum learning rate.                                                      |
| `device`     | 'cuda:0' | The device to run the model on (e.g., 'cuda:0' for GPU).                    |
| `beta`      | 2 | Beta values                                                                 |           
| `gamma`| 1.6     | Gamma values for the optimizer                                       |


### Datasets

[UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)

### Frameworks:
**Pytorch**





