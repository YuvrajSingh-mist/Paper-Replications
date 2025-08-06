# Conditional Generative Adversarial Networks (CGANs)

![CGAN Architecture](https://github.com/YuvrajSingh-mist/Paper-Replications/blob/main/CGANs/output_images/MNIST/fake_images_steps_14000.png)

## Overview

This repository contains a PyTorch implementation of Conditional Generative Adversarial Networks (CGANs) as described in the paper ["Conditional Generative Adversarial Nets" by Mirza & Osindero (2014)](https://arxiv.org/abs/1411.1784). CGANs extend the original GAN framework by conditioning both the generator and discriminator on auxiliary information, allowing for controlled generation of specific types of data.

## Key Features

- **Conditional Generation**: Generate MNIST digits conditioned on specific class labels (0-9)
- **Deep Convolutional Architecture**: Uses ConvTranspose2d layers for the generator and Conv2d layers for the discriminator
- **Label Embedding**: Efficient label representation using embedding layers
- **Instance Normalization**: Stable training with InstanceNorm2d layers
- **TensorBoard Logging**: Real-time monitoring of training progress and generated samples
- **Progressive Image Saving**: Saves generated images at regular intervals during training

## Architecture Details

### Generator
- **Input**: Random noise vector (100D) + class label
- **Embedding**: Class labels are embedded into 100D vectors
- **Architecture**: 
  - ConvTranspose2d layers with increasing spatial dimensions
  - InstanceNorm2d for stable training
  - ReLU activations (Tanh for output)
  - Output: 64x64 grayscale images

### Discriminator
- **Input**: 64x64 image + class label
- **Embedding**: Class labels embedded to match image dimensions
- **Architecture**:
  - Conv2d layers with decreasing spatial dimensions
  - InstanceNorm2d for stable training
  - LeakyReLU activations (Sigmoid for output)
  - Output: Binary classification (real/fake)

## Model Configuration

```python
@dataclass
class ModelArgs:
    latent_vector_size = 100      # Noise vector dimension
    batch_size = 128              # Training batch size
    num_classes = 10              # Number of MNIST classes
    img_size = 64                 # Output image size
    no_of_channels = 1            # Grayscale images
    dropout = 0.5                 # Dropout rate
    initial_lr = 0.1              # Initial learning rate
    final_lr = 1e-6               # Final learning rate
    momentum_initial = 0.5        # Initial momentum
    final_momentum_value = 0.7    # Final momentum
```

## Training Details

- **Dataset**: MNIST (28x28 → resized to 64x64)
- **Optimizer**: Adam with β₁=0.5, β₂=0.999
- **Learning Rate**: 0.0002
- **Loss Function**: Binary Cross-Entropy Loss
- **Epochs**: 30
- **Batch Size**: 128
- **Image Normalization**: [-1, 1] range using transforms.Normalize((0.5,), (0.5,))

## File Structure

```
CGANs/
├── cgan.ipynb              # Main implementation notebook
├── output_images/          # Generated images during training
│   └── MNIST/             # MNIST-specific outputs
│       ├── fake_images_steps_*.png    # Generated images at different steps
│       └── real_images_steps_*.png    # Real images for comparison
└── logs/                   # TensorBoard logs
    ├── fake/              # Fake image logs
    └── real/              # Real image logs
```

## Usage

### Training the Model

1. **Setup Environment**:
   ```python
   import torch
   import torchvision
   from torch import nn
   from torchvision import transforms
   from torch.utils.tensorboard import SummaryWriter
   ```

2. **Load Data**:
   ```python
   transforms = torchvision.transforms.Compose([
       transforms.Resize(size=(64, 64)),
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   
   trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                        download=True, transform=transforms)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
   ```

3. **Initialize Models**:
   ```python
   generator = Generator().to(device)
   discriminator = Discriminator().to(device)
   
   # Apply weight initialization
   generator.apply(weights_init)
   discriminator.apply(weights_init)
   ```

4. **Run Training**:
   Simply execute all cells in the `cgan.ipynb` notebook.

### Generating Specific Digits

```python
# Generate digit '7' examples
target_label = 7
noise = torch.randn(batch_size, 100, 1, 1, device=device)
labels = torch.full((batch_size,), target_label, device=device)

with torch.no_grad():
    fake_images = generator(noise, labels)
```

## Training Progress

The model saves generated images every 500 iterations, allowing you to monitor the quality improvement over time:

- **Early Training** (steps 0-1000): Noisy, unclear digit shapes
- **Mid Training** (steps 5000-8000): Recognizable digit structures emerge
- **Late Training** (steps 12000+): High-quality, diverse digit generation

## Key Implementation Details

### Label Conditioning
- **Generator**: Labels are embedded and concatenated with noise in feature space
- **Discriminator**: Labels are embedded to image dimensions and concatenated with input images

### Weight Initialization
```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
```

### Loss Functions
- **Generator Loss**: Tries to fool discriminator → `BCE(D(G(z,c)), 1)`
- **Discriminator Loss**: Real detection + Fake detection → `BCE(D(x,c), 1) + BCE(D(G(z,c)), 0)`

## Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir=logs
```

## Results

The trained CGAN successfully generates high-quality MNIST digits conditioned on specific class labels. The model demonstrates:

- **Class Consistency**: Generated images match the requested digit class
- **Diversity**: Multiple variations of each digit class
- **Quality**: Clear, recognizable handwritten digits
- **Stability**: Consistent performance across different random seeds

## Paper Reference

```bibtex
@article{mirza2014conditional,
  title={Conditional generative adversarial nets},
  author={Mirza, Mehdi and Osindero, Simon},
  journal={arXiv preprint arXiv:1411.1784},
  year={2014}
}
```

## Requirements

- PyTorch
- torchvision
- tensorboard
- torchinfo
- numpy
- matplotlib

## Future Enhancements

- [ ] Multi-class conditioning beyond MNIST
- [ ] Progressive growing for higher resolution outputs
- [ ] Spectral normalization for training stability
- [ ] FID/IS metrics for quantitative evaluation
- [ ] Conditional interpolation between classes
