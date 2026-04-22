import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# 1. Apply PyTorch memory fragmentation fix (suggested by your error message)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. Load the pipeline
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)

# 3. USE SEQUENTIAL OFFLOAD (The biggest VRAM saver)
# This replaces `enable_model_cpu_offload()`. Instead of moving whole models to the GPU, 
# it moves them layer-by-layer. It will be slower, but it uses drastically less VRAM.
pipe.enable_sequential_cpu_offload()

# 4. Enable VAE Memory Optimizations
# The VAE (which decodes the final image) is notorious for causing OOM errors at the very end.
if hasattr(pipe, "vae") and pipe.vae is not None:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

# 5. Load LoRA
pipe.load_lora_weights("prithivMLmods/Qwen-Image-Edit-2511-Object-Remover")

# 6. Load Image
input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

# 7. Run Inference
# Note: I recommend keeping the image size controlled. 
# If it's a massive 4K image, it will still crash. 
# The cat.png you linked is small enough, but keep it in mind for your own images.
print("Starting generation... (this may take a bit longer due to sequential offloading)")
image = pipe(
    prompt="Remove the cat from the image.",
    image=input_image
).images[0]

image.save("cleaned_test_output.png")
print("Saved successfully!")