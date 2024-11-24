import torch
from torch import nn
from torch.nn import functional as F
from CLIPTextEncoder import CLIP
from VAEncoder import VAE_Encoder
from VADecoder import VAE_Decoder
from diffusion import Diffusion
from CLIP_trainLogic import CLIPTrainer
from VAE import VanillaVAE

import converter_to_load

# def load_models_from_weights(ckpt_path, device):
#     state_dict = converter_to_load.load_from_standard_weights(ckpt_path, device)

#     encoder = VAE_Encoder().to(device)
#     encoder.load_state_dict(state_dict['encoder'], strict=True)

#     decoder = VAE_Decoder().to(device)
#     decoder.load_state_dict(state_dict['decoder'], strict=True)
    
#     # vae = VanillaVAE().to(device)
#     # # Load the checkpoint
#     # checkpoint = torch.load('pretrained_models/vae_model_epoch_600.pth', map_location=torch.device('cpu'))  
#     # vae.load_state_dict(checkpoint['model_state_dict'])


#     diffusion = Diffusion().to(device)
#     diffusion.load_state_dict(state_dict['diffusion'], strict=True)

#     clip_trainer = CLIPTrainer().to(device)
#     # clip.load_state_dict(state_dict['clip'], strict=True)
    
#     # Load the checkpoint
#     checkpoint = torch.load('pretrained_models/clip_epoch_81.pth', map_location=torch.device('cpu'))
#     clip_trainer.load_state_dict(checkpoint['model_state_dict'])
    
#     # clip = CLIP().to(device)
#     # clip.load_state_dict(state_dict['clip'], strict=True)


    

#     return {
#         'clip': clip_trainer.text_encoder,
#         'encoder': encoder,
#         'decoder': decoder,
#         'diffusion': diffusion,
#     }

def load_models_from_weights(ckpt_path, device):
    state_dict = converter_to_load.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }