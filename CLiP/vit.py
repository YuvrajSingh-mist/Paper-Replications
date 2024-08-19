import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision
import torchinfo
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

#Transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
class PatchEmbeddings(nn.Module):
    
    def __init__(self, 
                 in_channels: int=3, 
                 embeddings_dimensions: int=768,
                 patch_size: int=16):
        super().__init__()
        
        self.patch_size = patch_size
        self.patched_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=embeddings_dimensions, stride=patch_size, padding=0, kernel_size=patch_size)
        
        self.flatten_embeddings = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patched_embeddings(x)
        x_flatten = self.flatten_embeddings(x_patched)
        return x_flatten.permute(0,2,1)
    
    
class MultiHeadSelfAttentionBlock(nn.Module):
    
    def __init__(
        self,
        num_heads: int=12,
        embeddings_dimension: int=768,
        attn_dropout: int=0
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(embeddings_dimension)
        self.multihead_attn_layer = nn.MultiheadAttention(embed_dim=embeddings_dimension, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _  = self.multihead_attn_layer(query=x, key=x, value=x, need_weights=False)
        # print(attn_output)
        return attn_output
        
        
        
class MLPBlock(nn.Module):
  def __init__(self,
               embeddings_dimension:int=768,
               mlp_size:int=3072,
               dropout:int=0.1):
    super().__init__()
  

    self.layer_norm = nn.LayerNorm(normalized_shape=embeddings_dimension)


    self.mlp = nn.Sequential(
        nn.Linear(in_features=embeddings_dimension,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embeddings_dimension),
        nn.Dropout(p=dropout) 
    )
  
  def forward(self, x):
    x = self.layer_norm(x) 
    x = self.mlp(x)
    return x
    # return self.mlp(self.layer_norm(x)) # same as above 
    

class TransfornmerEncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int=12,
        embeddings_dimension: int=768,
        dropout: int=0.1,
        mlp_size:  int=3072,
        attn_dropout: int=0
    ):
        super().__init__()
        
        self.msa_layer = MultiHeadSelfAttentionBlock(num_heads=num_heads, embeddings_dimension=embeddings_dimension, attn_dropout=attn_dropout)
        
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_dimension=embeddings_dimension, mlp_size=mlp_size)
        
    def forward(self, x):
        x = self.msa_layer(x) + x
        x = self.mlp_block(x) + x
        
        return x



class ViT(nn.Module):
    def __init__(
        self,
        num_heads: int=12,
        embeddings_dimension: int=768,
        dropout: int=0.1,
        mlp_size:  int=3072,
        attn_dropout: int=0,
        num_of_encoder_layers: int=12,
        patch_size: int=16,
        image_width: int=224,
        img_height: int=224,
        no_channels: int=3,
        classes: int=1000,
        positional_embedding_dropout: int=0.1,
        projection_dims: int = 768
        
    ):
        assert (img_height * image_width) % patch_size == 0
        
        super().__init__()
        self.number_of_patches = (image_width * img_height)//(patch_size * patch_size)
        # print(self.number_of_patches
        self.patch_embeddings = PatchEmbeddings(in_channels=no_channels, embeddings_dimensions=embeddings_dimension,patch_size=patch_size)
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.number_of_patches + 1, embeddings_dimension), requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1,1, embeddings_dimension), requires_grad=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=embeddings_dimension)
        # self.encoder_layer = TransfornmerEncoderBlock(num_heads=num_heads, embeddings_dimension=embeddings_dimension, dropout=dropout, mlp_size=mlp_size,attn_dropout=attn_dropout)
        
        self.encoder_block = nn.Sequential(*[TransfornmerEncoderBlock(num_heads=num_heads, embeddings_dimension=embeddings_dimension, dropout=dropout, mlp_size=mlp_size,attn_dropout=attn_dropout) for _ in range(num_of_encoder_layers)])
        
        self.classifier = nn.Sequential(
            
            nn.LayerNorm(embeddings_dimension),
            nn.Linear(in_features=embeddings_dimension, out_features=projection_dims)
        )
        # self.multimodalVisionLayerProjector = nn.Linear(in_features=embeddings_dimension, out_features=projection_dims, device=ModelArgs.device)
        self.dropout_after_positional_embeddings = nn.Dropout(p=positional_embedding_dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embeddings(x)
        prepend_token = self.cls_token.expand(batch_size, -1, -1) 
        
        x = torch.cat((prepend_token, x), dim=1)
        x = self.positional_embeddings + x
        x = self.dropout_after_positional_embeddings(x)
        x = self.layer_norm(x)
        x = self.encoder_block(x)
        # x = self.multimodalVisionLayerProjector(x)
        x = self.classifier(x[:,0])
        
        return x
            
            
        
        
        


     