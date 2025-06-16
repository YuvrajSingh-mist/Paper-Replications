# Install required packages
# !pip install tqdm wandb torchinfo

# Core imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    
    
    
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
from torch.utils.data import random_split

inference = True

class labeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))  # More specific pattern for CelebA
        if not self.image_paths:
            # Fallback to general pattern if no .jpg files found
            self.image_paths = glob.glob(os.path.join(root_dir, '*'))
        
        self.transform = transform
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in directory: {root_dir}")
        
        self.image_paths.sort()  # Ensure consistent order of images
        # Extract base filenames for matching with labels
        self.base_names = [os.path.basename(path) for path in self.image_paths]
        
        # Load labels from CSV
        label_path = os.path.join('/speech/advait/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/', 'list_attr_celeba.csv')
        self.labels = pd.read_csv(label_path)
        
        # Map attributes to images efficiently
        # Create a dictionary for faster lookup
        attrs_dict = {}
        for _, row in self.labels.iterrows():
            filename = row.iloc[0]  # First column contains filenames
            attributes = row.iloc[1:].values  # All other columns are attributes
            attrs_dict[filename] = attributes
        
        # Prepare default attributes vector once
        default_attrs = np.zeros(len(self.labels.columns) - 1)
        
        # Use list comprehension with dictionary lookup for speed
        self.attrs = [attrs_dict.get(name, default_attrs) for name in self.base_names]
        
        # Convert attributes to tensor in one go
        self.attrs = torch.tensor(self.attrs, dtype=torch.float)
        print(f"Loaded {len(self.attrs)} attribute sets efficiently")
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.attrs[idx]  # Return both image and attributes
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image and zeros as fallback
            # if self.transform:
            #     return self.transform(Image.new('RGB', (128, 128), color='black')), torch.zeros_like(self.attrs[0])
            # else:


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

if(inference == False):
        print("Creating dataset...")
        dataset = UnlabeledImageDataset(image_dir, transform=transform)

        # Split 80% train, 20% val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        print(f"Splitting dataset: {train_size} train, {val_size} validation")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # DataLoaders with better settings for CelebA
        batch_size = 32
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
    def encoding_function(self, x):
        encoded = self.encoder(x)
        sampled_z = self.z_mean(encoded)
        log_var_sampled_z = self.z_log_var(encoded)
        z = self.reparametrize(encoded, sampled_z, log_var_sampled_z)
        return z, sampled_z, log_var_sampled_z
    def forward(self, x):
        # encoded = self.encoder(x)
        # sampled_z = self.z_mean(encoded)
        # log_var_sampled_z = self.z_log_var(encoded)
        # z = self.reparametrize(encoded, sampled_z, log_var_sampled_z)
        z, sampled_z, log_var_sampled_z = self.encoding_function(x)
        
        decoded = self.decoder(z)
        return decoded, sampled_z, log_var_sampled_z, z
    
    
autoencoder = Autoencoder(input_dim=3, hidden_dim=128, output_dim=64).to(device)

# Print model summary
from torchinfo import summary
print("Model Summary:")
print(summary(autoencoder, (32, 3, 128, 128), device=device))

weights = torch.load('./vae_checkpoint_epoch_240.pth', map_location=device)
autoencoder.load_state_dict(weights['model_state_dict'])


def plot():

    # Get a batch of validation data for visualization
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            res, mu, log_var, z = autoencoder(data)  # Unpack VAE outputs
            # break

    # Move tensors to CPU and convert to numpy for visualization
    original_images = data.cpu().numpy()
    reconstructed_images = res.cpu().numpy()

    # Plot original vs reconstructed images
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    fig.suptitle('VAE: Original (top) vs Reconstructed (bottom)', fontsize=16)

    for i in range(8):
        # Original images - transpose from (C, H, W) to (H, W, C) for RGB display
        orig_img = np.transpose(original_images[i], (1, 2, 0))
        orig_img = np.clip(orig_img, 0, 1)  # Ensure values are in [0, 1] range
        
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed images - transpose from (C, H, W) to (H, W, C) for RGB display
        recon_img = np.transpose(reconstructed_images[i], (1, 2, 0))
        recon_img = np.clip(recon_img, 0, 1)  # Ensure values are in [0, 1] range
        
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Log sample images to wandb
    wandb.log({
        "sample_reconstructions": wandb.Image(plt)
    })

    print("Visualization complete!")


#Latent space arithmetic

"""Targets are 40-dim vectors representing
    00 - 5_o_Clock_Shadow
    01 - Arched_Eyebrows
    02 - Attractive 
    03 - Bags_Under_Eyes
    04 - Bald
    05 - Bangs
    06 - Big_Lips
    07 - Big_Nose
    08 - Black_Hair
    09 - Blond_Hair
    10 - Blurry 
    11 - Brown_Hair 
    12 - Bushy_Eyebrows 
    13 - Chubby 
    14 - Double_Chin 
    15 - Eyeglasses 
    16 - Goatee 
    17 - Gray_Hair 
    18 - Heavy_Makeup 
    19 - High_Cheekbones 
    20 - Male 
    21 - Mouth_Slightly_Open 
    22 - Mustache 
    23 - Narrow_Eyes 
    24 - No_Beard 
    25 - Oval_Face 
    26 - Pale_Skin 
    27 - Pointy_Nose 
    28 - Receding_Hairline 
    29 - Rosy_Cheeks 
    30 - Sideburns 
    31 - Smiling 
    32 - Straight_Hair 
    33 - Wavy_Hair 
    34 - Wearing_Earrings 
    35 - Wearing_Hat 
    36 - Wearing_Lipstick 
    37 - Wearing_Necklace 
    38 - Wearing_Necktie 
    39 - Young         
"""
    
def compute_avg_score(data_loader, idx):
    with_feature_samples = []
    without_feature_samples = []
    
    # Process all batches
    for imgs, attrs in data_loader:
        imgs = imgs.to(device)
        attrs = attrs.to(device)
        
        # Filter images with the feature (attrs[:, idx] > 0)
        with_feature_mask = attrs[:, idx] > 0
        without_feature_mask = attrs[:, idx] <= 0
        
        # Get latent representations for images with the feature
        if with_feature_mask.any():
            with_feature_imgs = imgs[with_feature_mask]
            # print(with_feature_imgs.shape)
            with torch.inference_mode():
                # autoencoder.eval()
                with_feature_encoded, _, _ = autoencoder.encoding_function(with_feature_imgs)
                with_feature_samples.append(with_feature_encoded)
        
        # Get latent representations for images without the feature
        if without_feature_mask.any():
            with torch.inference_mode():
                # autoencoder.eval()
                without_feature_imgs = imgs[without_feature_mask]
                without_feature_encoded, _, _ = autoencoder.encoding_function(without_feature_imgs)
                without_feature_samples.append(without_feature_encoded)
    print(f"Found {len(with_feature_samples)} samples with feature at index {idx}")
    print(f"Found {len(without_feature_samples)} samples without feature at index {idx}")
    # Concatenate all samples and compute averages
    if with_feature_samples:
        with_feature = torch.cat(with_feature_samples, dim=0)
        avg_with_feature = with_feature.mean(dim=0)
    else:
        raise ValueError(f"No samples found with feature at index {idx}")
    
    if without_feature_samples:
        without_feature = torch.cat(without_feature_samples, dim=0)
        avg_without_feature = without_feature.mean(dim=0)
    else:
        raise ValueError(f"No samples found without feature at index {idx}")
    
    return avg_with_feature, avg_without_feature


# Enhanced transforms with normalization for better training
transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    # Normalize to [-1, 1] range (optional, but often helps with VAE training)
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Creating dataset...")
dataset = labeledImageDataset(image_dir, transform=transform)

# Add a custom collate function to ensure proper tensor output
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels

# Split dataset into training and validation sets
val_size = int(0.1 * len(dataset))  # 10% for validation
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders with better settings for CelebA
batch_size = 4  # Adjust as needed
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2, 
    # pin_memory=True, 
    drop_last=True,
    collate_fn=custom_collate_fn  # Use our custom collate function
)

print(f"DataLoaders created:")
print(f"  Validation batches: {len(val_loader)}")


# Example usage
_lambda = 0.1  # Adjust this value as needed for the arithmetic operation
feature_index = 24  # Replace with the desired feature index (0-39)
avg_with_feature, avg_without_feature = compute_avg_score(val_loader, feature_index)
# Perform arithmetic in the latent space
latent_space_arithmetic = avg_with_feature - _lambda * avg_without_feature
# Decode the result
decoded_image = autoencoder.decoder(latent_space_arithmetic.unsqueeze(0))
# Convert to numpy for visualization
decoded_image_np = decoded_image.cpu().detach().numpy()[0]
# Plot the decoded image
plt.figure(figsize=(6, 6))
plt.imshow(np.transpose(decoded_image_np, (1, 2, 0)))
plt.title(f"Decoded Image for Feature Index {feature_index}")
plt.axis('off')
plt.show()
# Log the decoded image to wandb
wandb.log({
    "latent_space_arithmetic": wandb.Image(plt)
})
# Finish wandb run
wandb.finish()
