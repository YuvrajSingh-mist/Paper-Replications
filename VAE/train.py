# Install required packages
# !pip install tqdm wandb torchinfo

# Core imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Progress bar and logging
from tqdm import tqdm
import wandb

# Check if CUDA is available
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    
    

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
from torch.utils.data import random_split

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))  # More specific pattern for CelebA
        if not self.image_paths:
            # Fallback to general pattern if no .jpg files found
            self.image_paths = glob.glob(os.path.join(root_dir, '*'))
        
        self.transform = transform
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in directory: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (128, 128), color='black'))
            else:
                return Image.new('RGB', (128, 128), color='black')

# Path to folder where images are present
image_dir = '/speech/advait/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/img_align_celeba/img_align_celeba'

# Enhanced transforms with normalization for better training
transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    # Normalize to [-1, 1] range (optional, but often helps with VAE training)
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Creating dataset...")
dataset = UnlabeledImageDataset(image_dir, transform=transform)

# Split 80% train, 20% val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

print(f"Splitting dataset: {train_size} train, {val_size} validation")
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders with better settings for CelebA
batch_size = 32 * 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=2, pin_memory=True, drop_last=True)

print(f"DataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")

# Test the data loader
print("Testing data loader...")
for imgs in val_loader:
    print(f"Batch shape: {imgs.shape}")  # Should be (batch_size, 3, 128, 128)
    print(f"Data type: {imgs.dtype}")
    print(f"Value range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
    break

print("Data loading setup complete!")


# Initialize wandb
wandb.init(
    project="vae-mnist",
    config={
        "learning_rate": 0.0005,
        "epochs": 200,
        "batch_size": 64,
        "input_dim": 1,
        "hidden_dim": 'CelebA6',
        "latent_dim": 2,
        "dataset": "MNIST",
        "architecture": "Variational Autoencoder",
        "reconstruction_weight": 0.1,
        "kl_weight": 0.5
    }
)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, leaky = 0.1):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim  , hidden_dim * 2 , kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2 , kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2 , hidden_dim * 2 , kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2 , hidden_dim * 2 , kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2 , hidden_dim * 2 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.Flatten(),
        )
        # self.fc2 = nn.Linear(3136, output_dim)

    def forward(self, x):
        x = self.conv(x)
        # x = self.fc2(x)
        # x = nn.functional.sigmoid(x)
        return x
    
# class Decoder(nn.Module):


enc = Encoder(input_dim=3, hidden_dim=128, output_dim=32).to('cuda')
# x = torch.randn(1, 3, 128, 128).to('cuda')  # Example input tensor
# output = enc(x)
# print("Output shape:", output.shape)  # Should print the shape of the output tensor

from torchinfo import summary
# summary(enc, (1,3,128,128), device='cuda')  # Print the model summary

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, leaky = 0.1):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(output_dim, 16384)
        self.conv = nn.Sequential(
            
            # Reshape(-1, hidden_dim * 2, 16, 16),
            nn.ConvTranspose2d(input_dim, 64 , kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),

             nn.ConvTranspose2d(64 , 64 ,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            
            nn.ConvTranspose2d(64, 64 * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64 * 2, 64 * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
            nn.ConvTranspose2d( 64 * 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(leaky),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 256, 8, 8)
        # print(x.shape)
        x = self.conv(x)
        x = nn.functional.sigmoid(x)    
        return x
    
    
# summary(Decoder(input_dim=256, hidden_dim=8, output_dim=8).to('cuda'), (1, 8), device='cuda')  # Print the model summary


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, output_dim)
        self.decoder = Decoder(hidden_dim * 2, hidden_dim // 4 , output_dim)
        self.z_mean = nn.Linear(16384, output_dim, bias=False)
        self.z_log_var = nn.Linear(16384, output_dim, bias=False)
        
    def reparametrize(self, encoded, mean_sampled, log_var_sampled):
        # Use the same device as the input tensors
        device = mean_sampled.device
        epsilon = torch.randn(log_var_sampled.size(0), log_var_sampled.size(1), device=device)
        
        # Reparameterization trick: z = μ + σ * ε
        res = mean_sampled + torch.exp(log_var_sampled / 2.0) * epsilon
        return res
        
    def forward(self, x):
        encoded = self.encoder(x)
        sampled_z = self.z_mean(encoded)
        log_var_sampled_z = self.z_log_var(encoded)
        z = self.reparametrize(encoded, sampled_z, log_var_sampled_z)
        decoded = self.decoder(z)
        return decoded, sampled_z, log_var_sampled_z, z
    
    
autoencoder = Autoencoder(input_dim=3, hidden_dim=128, output_dim=64).to(device)

# Print model summary
from torchinfo import summary
print("Model Summary:")
print(summary(autoencoder, (32, 3, 128, 128), device=device))

# Count parameters
# total_params = sum(p.numel() for p in autoencoder.parameters())
# trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)

# print(f"\nTotal parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")
# print(f"Model initialized on device: {next(autoencoder.parameters()).device}")


# Optional: Save model checkpoints periodically
import os

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: epoch {epoch}, loss {loss}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {filepath}")
        return 0, float('inf')

# Example usage (uncomment to use):
# checkpoint_dir = "checkpoints"
# os.makedirs(checkpoint_dir, exist_ok=True)
# 
# # To save during training (add this inside the epoch loop):
# if (epoch + 1) % 10 == 0:  # Save every 10 epochs
#     checkpoint_path = os.path.join(checkpoint_dir, f"vae_checkpoint_epoch_{epoch+1}.pth")
#     save_checkpoint(autoencoder, optimizer, epoch+1, avg_train_loss, checkpoint_path)

print("Checkpoint utilities defined. Add checkpoint saving to training loop if needed.")


optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0005)
# Training the autoencoder
epochs = 240  # Number of epochs for training
step = 0
for epoch in range(epochs):
    # Training phase
    autoencoder.train()
    train_loss = 0.0
   
    train_recon_loss = 0.0
    train_kl_loss = 0.0
    num_batches = 0
    
    # Use tqdm to wrap the train_loader directly
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training', leave=False)
    
    for data in train_pbar:
        data = data.to(device)
        optimizer.zero_grad()
        
        output, mu, log_var, z = autoencoder(data)
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(output, data, reduction='none')
        recon_loss = recon_loss.view(output.size(0), -1).sum(dim=1)
        recon_loss = recon_loss.mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()
        
        # Total loss
        total_loss = kl_loss + 1.0 * recon_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        num_batches += 1
        step += 1
        
        # Update progress bar with current losses
        train_pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Recon': f'{recon_loss.item():.4f}',
            'KL': f'{kl_loss.item():.4f}'
        })
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "step": step,
            "train_loss": total_loss.item(),
            "train_reconstruction_loss": recon_loss.item(),
            "train_kl_loss": kl_loss.item(),
        })

        # if (epoch) %  == 0:  
    checkpoint_path = os.path.join('./', f"vae_checkpoint_epoch_{epoch+1}.pth")
    save_checkpoint(autoencoder, optimizer, step+1, train_loss, checkpoint_path)
    
    # Calculate average training losses
    # avg_train_loss = train_loss / num_batches
    # avg_train_recon = train_recon_loss / num_batches
    # avg_train_kl = train_kl_loss / num_batches
    
    # Validation phase
    autoencoder.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_kl_loss = 0.0
    val_batches = 0
    
    # Use tqdm for validation loop as well
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation', leave=False)
    
    with torch.no_grad():
        for data in val_pbar:
            data = data.to(device)
            output, mu, log_var, z = autoencoder(data)
            
            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(output, data, reduction='none')
            recon_loss = recon_loss.view(output.size(0), -1).sum(dim=1)
            recon_loss = recon_loss.mean()
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            kl_loss = kl_loss.mean() 
            
            # Total loss
            total_loss = kl_loss + 1.0 * recon_loss
            
            val_loss += total_loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            val_batches += 1
            
            # Update validation progress bar
            val_pbar.set_postfix({
                'Val Loss': f'{total_loss.item()}',
                'Val Recon': f'{recon_loss.item()}',
                'Val KL': f'{kl_loss.item()}'
            })

            wandb.log({
                # "epoch": epoch + 1,
                # "val_loss": avg_train_loss,
                # "val_reconstruction_loss": avg_train_recon,
                # "val_kl_loss": avg_train_kl,
                "val_loss": total_loss.item(),
                "val_reconstruction_loss": recon_loss.item(),
                "val_kl_loss": kl_loss.item()
            })


    # Calculate average validation losses
    # avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    # avg_val_recon = val_recon_loss / val_batches if val_batches > 0 else 0
    # avg_val_kl = val_kl_loss / val_batches if val_batches > 0 else 0
    
    # # Log epoch averages to wandb
   
    
    # print(f'Epoch {epoch+1}/{epochs}:')
    # print(f'  Train - Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}')
    # print(f'  Val   - Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}')
    # print('-' * 80)