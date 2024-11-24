import torch
from torch import nn
from torch.nn import functional as F
from VADecoder import VAE_AttentionBlock, VAE_ResidualBlock
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

{# IMAGE_SIZE = 128

# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Using 3 channels
# ])


# class VAE_Encoder(nn.Module):
#     def __init__(self):
#         super(VAE_Encoder, self).__init__()

#         # Initial convolution and residual layers
#         self.initial_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
#         self.residual_block1 = VAE_ResidualBlock(128, 128).to(device)
#         self.residual_block2 = VAE_ResidualBlock(128, 128).to(device)

#         # Downsampling and more residual blocks
#         self.downsample1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)
#         self.residual_block3 = VAE_ResidualBlock(128, 256).to(device)
#         self.residual_block4 = VAE_ResidualBlock(256, 256).to(device)

#         self.downsample2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)
#         self.residual_block5 = VAE_ResidualBlock(256, 512).to(device)
#         self.residual_block6 = VAE_ResidualBlock(512, 512).to(device)

#         self.downsample3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0)
#         self.residual_block7 = VAE_ResidualBlock(512, 512).to(device)
#         self.residual_block8 = VAE_ResidualBlock(512, 512).to(device)
#         self.residual_block9 = VAE_ResidualBlock(512, 512).to(device)

#         # Attention and normalization
#         self.attention = VAE_AttentionBlock(512).to(device)
#         self.norm = nn.GroupNorm(32, 512)
#         self.activation = nn.SiLU()

#         # Final convolutional layers
#         # Because the padding=1, it means the width and height will increase by 2
#         # Out_Height = In_Height + Padding_Top + Padding_Bottom
#         # Out_Width = In_Width + Padding_Left + Padding_Right
#         # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
#         # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8).
#         self.final_conv3x3 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
#         # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
#         self.final_conv1x1 = nn.Conv2d(8, 8, kernel_size=1, padding=0)

#     def forward(self, x, noise=0):
#         # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
#         x = self.initial_conv(x)
#         # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
#         x = self.residual_block1(x)
#         # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
#         x = self.residual_block2(x)

#         # Downsample and apply padding
#         x = F.pad(x, (0, 1, 0, 1))  # Pad on the right and bottom
#         # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
#         x = self.downsample1(x)
#         # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
#         x = self.residual_block3(x)
#          # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
#         x = self.residual_block4(x)

#         x = F.pad(x, (0, 1, 0, 1))
#         # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
#         x = self.downsample2(x)
#         # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
#         x = self.residual_block5(x)
#         # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
#         x = self.residual_block6(x)

#         x = F.pad(x, (0, 1, 0, 1))
#         # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
#         x = self.downsample3(x)
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
#         x = self.residual_block7(x)
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
#         x = self.residual_block8(x)
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
#         x = self.residual_block9(x)

#         # Apply attention, normalization, and activation
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
#         x = self.attention(x)
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
#         x = self.norm(x)
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
#         x = self.activation(x)

#         # Final convolutions
#         # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8).
#         x = self.final_conv3x3(x)
#         # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
#         x = self.final_conv1x1(x)

#         # Split into mean and log variance
#         # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
#         mean, log_variance = torch.chunk(x, 2, dim=1)      # we are having the log_variance instead of the variance

#         # Clamp the log variance
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
#         log_variance = torch.clamp(log_variance, -30, 20)
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
#         variance = log_variance.exp()
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
#         stdev = variance.sqrt()

#         # Apply noise for reparameterization trick
#         # Transform N(0, 1) -> N(mean, stdev)
#         # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
#         x = mean + stdev * noise

#         # Scale the output
#         x *= 0.18215

#         return x,  mean, log_variance
}

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x