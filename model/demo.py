import model_loader
import Pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
model_file = "pretrained_models/v1-5-pruned-emaonly.ckpt"
models = model_loader.load_models_from_weights(model_file, DEVICE)

## TEXT TO IMAGE

prompt = "A cat without the glasses, and wearing comfy hat highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution"
# prompt = "A dragon soaring over a medieval castle with mountains in the background, during a thunderstorm with lightning in the sky."
uncond_prompt = "text, error, cropped, worst quality, low quality"  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
image_path = "images\cat.jpg"
input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.6

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = Pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer
)

# Combine the input image and the output image into a single image.
Image.fromarray(output_image).show()