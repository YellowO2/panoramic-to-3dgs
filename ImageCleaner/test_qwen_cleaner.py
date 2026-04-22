import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# 1. Automatically use Apple Silicon (MPS) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the base pipeline
# Using float16 instead of bfloat16 as float16 is much more stable on Apple Silicon (MPS)
print("Loading base model (this might take a while)...")
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.float16
)
# Move pipeline to the correct device
pipe = pipe.to(device)

# 3. Load the specific LoRA for object removal
print("Loading Object-Remover LoRA...")
pipe.load_lora_weights("prithivMLmods/Qwen-Image-Edit-2511-Object-Remover")

# 4. Prepare inference
# NOTE: Instead of a generic prompt, it's usually better to specify what to remove.
prompt = "Remove the cat from the image."

print("Fetching test image...")
input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

# 5. Run the model
print("Generating image...")
image = pipe(prompt=prompt, image=input_image).images[0]

# 6. Save the output so you can inspect it!
output_path = "cleaned_test_output.png"
image.save(output_path)
print(f"Done! Saved test result to {output_path}")
