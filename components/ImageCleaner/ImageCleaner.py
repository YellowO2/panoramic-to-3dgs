import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image


LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_WEIGHT_NAME = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_STEPS = 4
LIGHTNING_GUIDANCE = 1.0

REMOVE_LORA_REPO = "prithivMLmods/Qwen-Image-Edit-2511-Object-Remover"
REMOVE_DEFAULT_PROMPT = "Remove any people and vehicles."


class ImageCleaner:
    def __init__(
        self,
        model_id="Qwen/Qwen-Image-Edit-2511",
        device="cuda",
        offload=False,
    ):
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        if offload:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(device)

        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()

        self.pipe.load_lora_weights(
            LIGHTNING_REPO,
            weight_name=LIGHTNING_WEIGHT_NAME,
            adapter_name="speed",
        )
        self.pipe.load_lora_weights(REMOVE_LORA_REPO, adapter_name="remove")
        self.pipe.set_adapters("speed")

    def edit(self, image_path, prompt, mode="general", output_path=None):
        """Edit an image with a freeform prompt.

        mode:
          - "general": Lightning LoRA only (any prompt).
          - "remove_objects": Lightning + Object-Remover LoRAs stacked.
        """
        if mode == "remove_objects":
            self.pipe.set_adapters(["speed", "remove"], adapter_weights=[1.0, 1.0])
        else:
            self.pipe.set_adapters("speed")

        input_image = load_image(image_path)
        print(f"Editing image {image_path} (mode={mode})...")
        image = self.pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=LIGHTNING_STEPS,
            guidance_scale=LIGHTNING_GUIDANCE,
        ).images[0]

        if output_path:
            image.save(output_path)
            print(f"Saved edited image to {output_path}")
        return image

    def clean(self, image_path, prompt=REMOVE_DEFAULT_PROMPT, output_path=None):
        return self.edit(image_path, prompt, mode="remove_objects", output_path=output_path)
