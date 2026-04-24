import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

class ImageCleaner:
    def __init__(self, model_id="Qwen/Qwen-Image-Edit-2511", lora_weights="prithivMLmods/Qwen-Image-Edit-2511-Object-Remover", device="cuda"):
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        self.pipe.enable_sequential_cpu_offload()

        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()

        self.pipe.load_lora_weights(lora_weights)

    def clean(self, image_path, prompt="Remove any people and vehicles.", output_path=None):
        input_image = load_image(image_path)
        print(f"Cleaning image {image_path}...")
        image = self.pipe(
            prompt=prompt,
            image=input_image
        ).images[0]

        if output_path:
            image.save(output_path)
            print(f"Saved cleaned image to {output_path}")
            
        return image
