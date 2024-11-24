# import seaborn as sns
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from CLIPTextEncoder import CLIP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
def the_tokenizer(text):
        tokens = def_tokenizer(
            text, 
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return tokens["input_ids"].squeeze()  # shape: (context_length,)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_heads=12, embed_dim=768, depth=12):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Create patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) for _ in range(depth)
        ])
        
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch_size, in_channels, height, width)
        batch_size = x.shape[0]
        x = self.patch_embedding(x)  # (batch_size, embed_dim, num_patches_height, num_patches_width)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        class_token = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((class_token, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x += self.position_embedding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Final layer normalization
        x = self.layernorm(x)
        
        return x   # Should output (batchsize, num_patches + 1, embed_dim)



class CLIPTrainer(nn.Module):
    def __init__(self, num_epochs: int = 25):
        super(CLIPTrainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the text and image encoders
        self.text_encoder = CLIP().to(self.device) 
        self.image_encoder = VisionTransformer().to(self.device) 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 
        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(), lr=5e-5
        )
        self.num_epochs = num_epochs


    def forward(self, text: torch.Tensor, images: torch.Tensor):
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(images)
        
        # Normalize features for better performance
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        return text_features, image_features

    def train_model(self, train_loader: DataLoader) -> List[float]:
        self.to(self.device)
        loss_history = []
        accumulation_steps = 64  # for effective batch size (e.g., 64 * 8 = 512)
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            # Calculate the logit scale once per epoch
#             logit_scale = self.logit_scale.exp()


            # Add progress bar for batch iteration
            for batch_idx, (images, texts) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)):
                images, texts = images.to(self.device), texts.to(self.device)

                # Zero gradients at start of accumulation step
                if batch_idx % accumulation_steps == 0:
                    self.optimizer.zero_grad()
                
                
                # Forward pass (Batch_Size, Seq_Len, Dim) and (batchsize, num_patches = 197, embed_dim)
                text_features, image_features = self.forward(texts, images)
                
#                 print("image_features  ", image_features.shape)
#                 print("text_features  ", text_features.shape)
                
                # (batchsize, num_patches = 197, embed_dim) -> (batchsize,  embed_dim)  [mean] -> (batchsize, embed_dim)
                image_features_agg = F.normalize(image_features.mean(dim=1), dim=-1)
                text_features_agg = F.normalize(text_features.mean(dim=1), dim=-1)
                
#                 print("image_features_agg  ", image_features_agg.shape)
#                 print("text_features_agg  ", text_features_agg.shape)

                # Compute similarity matrix
                logits_per_image =   self.logit_scale.exp() * torch.matmul(image_features_agg, text_features_agg.T)   # (batch_size * batch_size)
#                 print("logits_per_image  ", logits_per_image.shape)
                
                logits_per_text = logits_per_image.T
#                 print("logits_per_text  ", logits_per_text.shape)

                # Label: Each image-caption pair in the batch is treated as a positive pair
                # labels => (batchsize)
                labels = torch.arange(len(image_features)).to(self.device)
#                 print("lables ", labels.shape)

                # Cross-entropy loss in both directions
                image_to_text_loss = F.cross_entropy(logits_per_image, labels)
                text_to_image_loss = F.cross_entropy(logits_per_text, labels)

                # Total contrastive loss
                loss = (image_to_text_loss + text_to_image_loss) / 2
                loss.backward(retain_graph=True)
                
                # Step optimizer only after full accumulation steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()  # Clear gradients for next accumulation
                    
                    
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss}")

            # Save checkpoint for each epoch
            checkpoint_path = f"clip_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        return loss_history

    def resume_training(self, train_loader: DataLoader, checkpoint_path: str, new_lr: float = None) -> List[float]:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_history = [checkpoint['loss']]
        

            # Update learning rate if a new one is provided
        if new_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to {new_lr}")
    
        # Set model to the correct device
        self.to(self.device)
    
        # Ensure optimizer tensors are on the same device as the model
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
    
        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = 0.0
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
    
            for batch_idx, (images, texts) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)):
                images, texts = images.to(self.device), texts.to(self.device)
    
                if batch_idx % 64 == 0:
                    self.optimizer.zero_grad()
    
                # Forward pass
                text_features, image_features = self.forward(texts, images)
                image_features_agg = F.normalize(image_features.mean(dim=1), dim=-1)
                text_features_agg = F.normalize(text_features.mean(dim=1), dim=-1)
                logits_per_image = self.logit_scale.exp() * torch.matmul(image_features_agg, text_features_agg.T)
                logits_per_text = logits_per_image.T
                labels = torch.arange(len(image_features)).to(self.device)
    
                # Contrastive loss
                image_to_text_loss = F.cross_entropy(logits_per_image, labels)
                text_to_image_loss = F.cross_entropy(logits_per_text, labels)
                loss = (image_to_text_loss + text_to_image_loss) / 2
                loss.backward(retain_graph=True)
    
                if (batch_idx + 1) % 64 == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
    
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss}")
    
            # Save checkpoint for each epoch
            checkpoint_path = f"clip_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
        return loss_history


    