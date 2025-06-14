# Variational Autoencoder (VAE) from Scratch

I implemented a Variational Autoencoder Architecture from Scratch using PyTorch on MNIST dataset.

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Results

### Original vs Reconstructed Images

The following image shows the comparison between original MNIST digits (top row) and their reconstructions by the VAE (bottom row):

![VAE Results](vae_results.png)

*VAE: Original (top) vs Reconstructed (bottom) - Shows the model's ability to reconstruct handwritten digits from the latent space representation.*

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
- Final linear layers for mean and log variance (reparameterization trick)

**Decoder**:
- Linear layer to expand latent representation
- 4 Transposed Convolutional layers with LeakyReLU activation
- Sigmoid activation for final output

### Training Details

**Optimizer**: Adam with learning rate 0.0005  
**Loss Function**: Reconstruction Loss (MSE) + KL Divergence  
**Training/Validation Split**: 80/20  
**Device**: CUDA

### VAE-Specific Components

**Reparameterization Trick**: Enables backpropagation through stochastic sampling  
**KL Divergence**: Regularizes latent space to follow standard normal distribution  
**Latent Space**: 2D for visualization and compression

### Results

The VAE successfully reconstructs MNIST digits while learning a meaningful 2-dimensional latent space. The model balances reconstruction quality with latent space regularization through the KL divergence term.

**Final Training Metrics**:
- Reconstruction Loss: Measures how well images are reconstructed
- KL Loss: Ensures latent variables follow desired distribution
- Total Loss: Combination of both terms

### Frameworks:
**PyTorch**
