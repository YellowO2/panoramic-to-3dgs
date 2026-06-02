import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image


LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_WEIGHT_NAME = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_STEPS = 4
LIGHTNING_GUIDANCE = 1.0

REMOVE_LORA_REPO = "prithivMLmods/Qwen-Image-Edit-2511-Object-Remover"
REMOVE_DEFAULT_PROMPT = "Remove any people and vehicles."

# Qwen-Image-Edit-2511 OOMs much past ~2MP. We pick the largest aspect-preserving
# size under this budget, rounded to a multiple of 16 (VAE + patch stride).
MAX_EDIT_PIXELS = 2048 * 1024
EDIT_SIZE_MULTIPLE = 16


def _fit_edit_size(w: int, h: int) -> tuple[int, int]:
    aspect = w / h
    target_h = int((MAX_EDIT_PIXELS / aspect) ** 0.5)
    target_w = int(target_h * aspect)
    target_w = max(EDIT_SIZE_MULTIPLE, (target_w // EDIT_SIZE_MULTIPLE) * EDIT_SIZE_MULTIPLE)
    target_h = max(EDIT_SIZE_MULTIPLE, (target_h // EDIT_SIZE_MULTIPLE) * EDIT_SIZE_MULTIPLE)
    return target_w, target_h


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
        orig_w, orig_h = input_image.size
        edit_w, edit_h = _fit_edit_size(orig_w, orig_h)
        print(f"Editing {image_path} (mode={mode}, edit_size={edit_w}x{edit_h}, restored to {orig_w}x{orig_h})...")
        image = self.pipe(
            prompt=prompt,
            image=input_image,
            width=edit_w,
            height=edit_h,
            num_inference_steps=LIGHTNING_STEPS,
            guidance_scale=LIGHTNING_GUIDANCE,
        ).images[0]

        if (image.width, image.height) != (orig_w, orig_h):
            image = image.resize((orig_w, orig_h), Image.LANCZOS)

        if output_path:
            image.save(output_path)
            print(f"Saved edited image to {output_path}")
        return image

    def clean(self, image_path, prompt=REMOVE_DEFAULT_PROMPT, output_path=None):
        return self.edit(image_path, prompt, mode="remove_objects", output_path=output_path)
