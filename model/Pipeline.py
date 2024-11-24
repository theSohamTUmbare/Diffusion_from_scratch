import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from ddpm import DDPMSampler
from transformers import CLIPTokenizer

{
# def_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
# def the_tokenizer(text):
#         tokens = def_tokenizer(
#             text, 
#             max_length=77,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         return tokens["input_ids"].squeeze()  # shape: (context_length,)


# WIDTH = 128
# HEIGHT = 128
# LATENTS_WIDTH = WIDTH // 8
# LATENTS_HEIGHT = HEIGHT // 8

# def generate(
#     prompt, uncond_prompt=None,
#     input_image = None,
#     strength=0.8, 
#     do_cfg=True, 
#     cfg_scale=7.5,  ## its the weight to how much we wannt to pay attention to the condition
#     sampler_name="ddpm",
#     n_inference_steps=50,
#     models={},
#     seed=None,
#     device=None, idle_device=None, 
#     tokenizer=None,
#     input_image_path=None,
# ):
#     with torch.no_grad():
#         if not 0 < strength <= 1:
#             raise ValueError("strength must be between 0 and 1")

#         if idle_device:
#             to_idle = lambda x: x.to(idle_device)
#         else:
#             to_idle = lambda x: x

#         # Initialize random number generator according to the seed specified
#         generator = torch.Generator(device=device)
#         if seed is None:
#             generator.seed()
#         else:
#             generator.manual_seed(seed)

#         clip = models["clip"]
#         clip.to(device)
        
#         if do_cfg:
#             # Convert into a list of length Seq_Len=77
#             # cond_tokens = tokenizer.batch_encode_plus(
#             #     [prompt], padding="max_length", max_length=77
#             # ).input_ids
            
#             cond_tokens = the_tokenizer(prompt).unsqueeze(0)
            
#             # (Batch_Size, Seq_Len)
#             cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            
#             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
#             cond_context = clip(cond_tokens)
            
#             # uncond_tokens = tokenizer.batch_encode_plus(
#             #     [uncond_prompt], padding="max_length", max_length=77
#             # ).input_ids

#             # Convert into a list of length Seq_Len=77
#             uncond_tokens = the_tokenizer(uncond_prompt).unsqueeze(0)
            
#             # (Batch_Size, Seq_Len)
#             uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            
#             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
#             uncond_context = clip(uncond_tokens)
            
#             # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
#             context = torch.cat([cond_context, uncond_context])
        
#         else:
            
#             # tokens = tokenizer.batch_encode_plus(
#             #     [prompt], padding="max_length", max_length=77
#             # ).input_ids
            
#             # Convert into a list of length Seq_Len=77
#             tokens = the_tokenizer(prompt).unsqueeze(0)
            
#             # (Batch_Size, Seq_Len)
#             tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            
#             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
#             context = clip(tokens)
        
#         to_idle(clip)  ## put  clip to the cpu to have more space on gpu

#         if sampler_name == "ddpm":
#             sampler = DDPMSampler(generator)
#             sampler.set_inference_timesteps(n_inference_steps)
#         else:
#             raise ValueError("Unknown sampler value %s. ")

#         latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

#         if input_image_path:
#             encoder = models["encoder"]
#             encoder.to(device)
            
#             transform = transforms.Compose([
#                 transforms.Resize((WIDTH, HEIGHT)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Using 3 channels
#             ])



#             # input_image_tensor = input_image.resize((WIDTH, HEIGHT))
#             image = image.open(input_image_path)
            
#             # (Height, Width, Channel)
#             # input_image_tensor = np.array(input_image_tensor)

            
#             # (Height, Width, Channel) -> (Height, Width, Channel)
#             # input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            
#             # (Height, Width, Channel) -> (Height, Width, Channel)
#             input_image_tensor = rescale(input_image_tensor, (0, 255), (0, 1))
#             input_image_tensor = transform(image).to(device)
            
            
#             # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
#             input_image_tensor = input_image_tensor.unsqueeze(0)
            
#             # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
#             input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             encoder_noise = torch.randn(latents_shape, generator=generator, device=device)  ## Adding the noise to the encoder output
            
#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             latents = encoder(input_image_tensor, encoder_noise)   ## latent representation of the image with added noise

#             # Add noise to the latents (the encoded input image)
#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             sampler.set_strength(strength=strength) 
#             latents = sampler.add_noise(latents, sampler.timesteps[0])   # FORWARD DIFFUSION!

#             to_idle(encoder)    ## throw encoder to the cpu
        
#         else:
#             # For text-to-image model directly start with the random noise N(0, I)
#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             latents = torch.randn(latents_shape, generator=generator, device=device)

#         diffusion = models["diffusion"]
#         diffusion.to(device)

#         timesteps = tqdm(sampler.timesteps)   ## REVERSE DIFFUSION!
#         for i, timestep in enumerate(timesteps):
#             # (1, 320)
#             time_embedding = get_time_embedding(timestep).to(device)

#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             model_input = latents

#             if do_cfg:
#                 # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
#                 model_input = model_input.repeat(2, 1, 1, 1)

#             # model_output is the predicted noise
#             # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
#             model_output = diffusion(model_input, context, time_embedding)  # noise predicted by the unet which will be used to calculate then ued to caculate qt-1 
#             ##passing the latent of image with noise and the context which is the input text embeddings to crossattention with the latent image and perform unet thing
#             ## Do context free guidance
#             if do_cfg:
#                 output_cond, output_uncond = model_output.chunk(2)
#                 model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

#             # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
#             latents = sampler.step(timestep, latents, model_output)   ## gives qt-1

#         to_idle(diffusion)

#         decoder = models["decoder"]
#         decoder.to(device)
        
#         # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
#         images = decoder(latents)
        
#         to_idle(decoder)
        
#         # reverse_transform = transforms.Compose([
#         #     transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # Reverse normalization
#         #     # transforms.ToPILImage()  # Convert back to PIL image format
#         # ])
#         # images = reverse_transform(images.cpu())

#         images = rescale(images, (-1, 1), (0, 255), clamp=True)
#         # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
#         images = images.permute(0, 2, 3, 1)
#         images = images.to("cpu", torch.uint8).numpy()
#         return images[0]
    
# def rescale(x, old_range, new_range, clamp=False):
#     old_min, old_max = old_range
#     new_min, new_max = new_range
#     x -= old_min
#     x *= (new_max - new_min) / (old_max - old_min)
#     x += new_min
#     if clamp:
#         x = x.clamp(new_min, new_max)
#     return x

# def get_time_embedding(timestep):
#     # Shape: (160,)
#     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
#     # Shape: (1, 160)
#     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
#     # Shape: (1, 160 * 2)
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



"""for mae in india clip encoder"""
# def_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
# def the_tokenizer(text):
#         tokens = def_tokenizer(
#             text, 
#             max_length=77,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         return tokens["input_ids"].squeeze()  # shape: (context_length,)


# WIDTH = 128
# HEIGHT = 128
# LATENTS_WIDTH = WIDTH // 8
# LATENTS_HEIGHT = HEIGHT // 8

# def generate(
#     prompt, uncond_prompt=None,
#     input_image = None,
#     strength=0.8, 
#     do_cfg=True, 
#     cfg_scale=7.5,  ## its the weight to how much we wannt to pay attention to the condition
#     sampler_name="ddpm",
#     n_inference_steps=50,
#     models={},
#     seed=None,
#     device=None, idle_device=None, 
#     tokenizer=None,
#     input_image_path=None,
# ):
#     with torch.no_grad():
#         if not 0 < strength <= 1:
#             raise ValueError("strength must be between 0 and 1")

#         if idle_device:
#             to_idle = lambda x: x.to(idle_device)
#         else:
#             to_idle = lambda x: x

#         # Initialize random number generator according to the seed specified
#         generator = torch.Generator(device=device)
#         if seed is None:
#             generator.seed()
#         else:
#             generator.manual_seed(seed)

#         clip = models["clip"]
#         clip.to(device)
        
#         if do_cfg:
#             # Convert into a list of length Seq_Len=77
#             # cond_tokens = tokenizer.batch_encode_plus(
#             #     [prompt], padding="max_length", max_length=77
#             # ).input_ids
            
#             cond_tokens = the_tokenizer(prompt).unsqueeze(0)
            
#             # (Batch_Size, Seq_Len)
#             cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            
#             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
#             cond_context = clip(cond_tokens)
            
#             # uncond_tokens = tokenizer.batch_encode_plus(
#             #     [uncond_prompt], padding="max_length", max_length=77
#             # ).input_ids

#             # Convert into a list of length Seq_Len=77
#             uncond_tokens = the_tokenizer(uncond_prompt).unsqueeze(0)
            
#             # (Batch_Size, Seq_Len)
#             uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            
#             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
#             uncond_context = clip(uncond_tokens)
            
#             # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
#             context = torch.cat([cond_context, uncond_context])
        
#         else:
            
#             # tokens = tokenizer.batch_encode_plus(
#             #     [prompt], padding="max_length", max_length=77
#             # ).input_ids
            
#             # Convert into a list of length Seq_Len=77
#             tokens = the_tokenizer(prompt).unsqueeze(0)
            
#             # (Batch_Size, Seq_Len)
#             tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            
#             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
#             context = clip(tokens)
        
#         to_idle(clip)  ## put  clip to the cpu to have more space on gpu

#         if sampler_name == "ddpm":
#             sampler = DDPMSampler(generator)
#             sampler.set_inference_timesteps(n_inference_steps)
#         else:
#             raise ValueError("Unknown sampler value %s. ")

#         latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

#         if input_image:
#             encoder = models["encoder"]
#             encoder.to(device)

#             input_image_tensor = input_image.resize((WIDTH, HEIGHT))
#             # (Height, Width, Channel)
#             input_image_tensor = np.array(input_image_tensor)
#             # (Height, Width, Channel) -> (Height, Width, Channel)
#             input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
#             # (Height, Width, Channel) -> (Height, Width, Channel)
#             input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
#             # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
#             input_image_tensor = input_image_tensor.unsqueeze(0)
#             # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
#             input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             latents = encoder(input_image_tensor, encoder_noise)

#             # Add noise to the latents (the encoded input image)
#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             sampler.set_strength(strength=strength)
#             latents = sampler.add_noise(latents, sampler.timesteps[0])

#             to_idle(encoder)
#         else:
#             # For text-to-image model directly start with the random noise N(0, I)
#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             latents = torch.randn(latents_shape, generator=generator, device=device)

#         diffusion = models["diffusion"]
#         diffusion.to(device)

#         timesteps = tqdm(sampler.timesteps)   ## REVERSE DIFFUSION!
#         for i, timestep in enumerate(timesteps):
#             # (1, 320)
#             time_embedding = get_time_embedding(timestep).to(device)

#             # (Batch_Size, 4, Latents_Height, Latents_Width)
#             model_input = latents

#             if do_cfg:
#                 # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
#                 model_input = model_input.repeat(2, 1, 1, 1)

#             # model_output is the predicted noise
#             # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
#             model_output = diffusion(model_input, context, time_embedding)  # noise predicted by the unet which will be used to calculate then ued to caculate qt-1 
#             ##passing the latent of image with noise and the context which is the input text embeddings to crossattention with the latent image and perform unet thing
#             ## Do context free guidance
#             if do_cfg:
#                 output_cond, output_uncond = model_output.chunk(2)
#                 model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

#             # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
#             latents = sampler.step(timestep, latents, model_output)   ## gives qt-1

#         to_idle(diffusion)

#         decoder = models["decoder"]
#         decoder.to(device)
#         # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
#         images = decoder(latents)
#         to_idle(decoder)

#         images = rescale(images, (-1, 1), (0, 255), clamp=True)
#         # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
#         images = images.permute(0, 2, 3, 1)
#         images = images.to("cpu", torch.uint8).numpy()
#         return images[0]
    
# def rescale(x, old_range, new_range, clamp=False):
#     old_min, old_max = old_range
#     new_min, new_max = new_range
#     x -= old_min
#     x *= (new_max - new_min) / (old_max - old_min)
#     x += new_min
#     if clamp:
#         x = x.clamp(new_min, new_max)
#     return x

# def get_time_embedding(timestep):
#     # Shape: (160,)
#     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
#     # Shape: (1, 160)
#     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
#     # Shape: (1, 160 * 2)
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


}


import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from transformers import CLIPTokenizer
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)
        
        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)