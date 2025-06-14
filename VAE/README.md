# Autoencoder (AE) from Scratch

I implemented an Autoencoder Architecture from Scratch using Pytorch on MNIST dataset.

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## ModelArgs Hyperparameters

| Parameter      | Value | Description                                                                 
|----------------|-------|-----------------------------------------------------------------------------|
| `input_dim`    | 1     | Input channels (grayscale images).                                          |
| `hidden_dim`   | 64    | Hidden dimension for convolutional layers.                                  |
| `output_dim`   | 2     | Latent space dimension (bottleneck).                                       |
| `batch_size`   | 64    | The number of samples processed before the model is updated.                |
| `learning_rate`| 0.0005| Learning rate for Adam optimizer.                                          |
| `epochs`       | 200   | Number of training epochs.                                                  |
| `leaky_relu`   | 0.01  | Negative slope for LeakyReLU activation.                                   |

### Datasets

**MNIST**: Used torchvision to download the MNIST dataset directly (60,000 training images)

### Frameworks:
**Pytorch**

### Architecture

**Encoder**: 
- 4 Convolutional layers with LeakyReLU activation
- Stride 2 for downsampling in middle layers
- Final linear layer maps to latent space

**Decoder**:
- Linear layer to expand latent representation
- 4 Transposed Convolutional layers with LeakyReLU activation
- Sigmoid activation for final output

### Training Details

**Optimizer**: Adam with learning rate 0.0005  
**Loss Function**: Mean Squared Error (MSE)  
**Training/Validation Split**: 80/20  
**Device**: CUDA

### Results

The autoencoder successfully reconstructs MNIST digits with a 2-dimensional latent space compression. The model shows good reconstruction quality despite the extreme compression ratio (784 → 2 → 784).

**Final Training Loss**: Converged after 200 epochs

### Frameworks:
**Pytorch**
