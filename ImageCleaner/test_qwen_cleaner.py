import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.load_lora_weights("prithivMLmods/Qwen-Image-Edit-2511-Object-Remover")

input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

image = pipe(
    prompt="Remove the cat from the image.",
    image=input_image
).images[0]

image.save("cleaned_test_output.png")
