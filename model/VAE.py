import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor
from VAEncoder import VAE_Encoder
from VADecoder import VAE_Decoder
from VAE_trainLogic import vae_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VanillaVAE(nn.Module):

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, hidden_dims: List[int] = None, num_epochs: int = 25):    ## here in_channels and latent_dim are not useful at all model dont need thesee input these are there because i dont want to change the architecture for now
        """
        Initialize the VAE model with encoder and decoder architectures.
        """
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.encoder = VAE_Encoder().to(device)
        self.decoder = VAE_Decoder().to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.num_epochs = num_epochs
    
    def getEncoder(self, nk, x):
        nx, _, _ = self.encoder(x)
        return nx
        
    def forward(self, x):
        # pass the input through the encoder to get the latent vector
        z, z_mean, z_log_var = self.encoder(x)
        # pass the latent vector through the decoder to get the reconstructed
        # image
        reconstruction = self.decoder(z)
        # return the mean, log variance and the reconstructed image
        return z_mean, z_log_var, reconstruction

    def reparameterize(self, mu: Tensor, lv: Tensor) -> Tensor:  # lv is log_variance
        # print(f"mu-", mu.size())
        # print("lv-", lv.size())
        # print("eps-", epsilon.size())
        epsilon = torch.randn(mu.shape, device=mu.device)
        sigma = torch.exp(0.5 * lv)  # Standard deviation
        z = mu + sigma * epsilon
        return z

    def train(self, train_loader, start_epoch=0) -> List[Tensor]:
        self.to(device)

        for epoch in range(start_epoch, self.num_epochs):  # Start from start_epoch
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                pred = self(batch)
                loss = vae_loss(pred, batch, epoch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()  # Accumulate loss for average calculation
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

                # Save model and optimizer state at epoch 40 and every 49 epochs
                if (epoch + 1) == 40 or (epoch + 1) % 50 == 0:
                    torch.save({
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch + 1,
                    }, f"vae_model_epoch_{epoch + 1}.pth")
                    print(f"Model and optimizer state saved at epoch {epoch + 1}")

                if loss == 0:
                    print("stopped")
                    break

            # Optionally print the average loss for this epoch
            print(f"Epoch: {epoch + 1}, Avg_Loss: {epoch_loss / len(train_loader)}")
            print()

        return loss




