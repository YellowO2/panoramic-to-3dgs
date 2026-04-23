import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()

# The VAE (which decodes the final image) is notorious for causing OOM errors at the very end.
if hasattr(pipe, "vae") and pipe.vae is not None:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

# 5. Load LoRA
pipe.load_lora_weights("prithivMLmods/Qwen-Image-Edit-2511-Object-Remover")

# 6. Load Image
input_image = load_image(
    "/home/q1intern/panoramic-to-3dgs/round3.jpg"
)

# 7. Run Inference
# Note: I recommend keeping the image size controlled. 
# If it's a massive 4K image, it will still crash. 
# The cat.png you linked is small enough, but keep it in mind for your own images.
print("Start cleaning...")
image = pipe(
    prompt="Remove any people and vehicles.",
    image=input_image
).images[0]

image.save("cleaned_test_output.png")
print("Done!")