import numpy as np
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

# Equirectangular panoramas have ~2:1 aspect. Below this threshold we treat the
# image as a regular photo and skip chunking.
PANO_ASPECT_THRESHOLD = 1.8


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

    def _run_pipe(self, image, prompt):
        return self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=LIGHTNING_STEPS,
            guidance_scale=LIGHTNING_GUIDANCE,
        ).images[0]

    def _chunked_pano_edit(self, image, prompt):
        """Edit a 2:1 equirectangular pano by tiling into square chunks with
        50% overlap, editing each, and blending with a triangular alpha feather.
        The last chunk wraps the seam (right edge → left edge of source)."""
        W, H = image.size
        chunk_w = H  # square chunks (H × H)
        stride = chunk_w // 2
        n = max(1, W // stride)

        # Extract chunks, wrapping source pixels for chunks that cross the seam.
        chunks = []
        for i in range(n):
            x = i * stride
            if x + chunk_w <= W:
                c = image.crop((x, 0, x + chunk_w, H))
            else:
                left = image.crop((x, 0, W, H))
                right = image.crop((0, 0, (x + chunk_w) - W, H))
                c = Image.new("RGB", (chunk_w, H))
                c.paste(left, (0, 0))
                c.paste(right, (W - x, 0))
            chunks.append(c)

        # Edit each chunk.
        edited_arrs = []
        for i, c in enumerate(chunks):
            print(f"  pano chunk {i+1}/{n} editing...")
            out = self._run_pipe(c, prompt)
            if (out.width, out.height) != (chunk_w, H):
                out = out.resize((chunk_w, H), Image.LANCZOS)
            edited_arrs.append(np.array(out, dtype=np.float32))

        # Triangular feather: peak 1.0 at chunk center, ~0 at edges.
        w_1d = 1.0 - np.abs(2.0 * (np.arange(chunk_w) + 0.5) / chunk_w - 1.0)
        w_1d = np.clip(w_1d, 0.01, 1.0).astype(np.float32)
        w_2d = np.tile(w_1d, (H, 1))

        # Composite on an extended canvas (W + chunk_w wide), then fold the
        # right tail back into the left so wraparound chunks blend cleanly.
        ext_w = W + chunk_w
        acc = np.zeros((H, ext_w, 3), dtype=np.float32)
        wsum = np.zeros((H, ext_w), dtype=np.float32)
        for i, arr in enumerate(edited_arrs):
            x = i * stride
            acc[:, x:x + chunk_w, :] += arr * w_2d[:, :, None]
            wsum[:, x:x + chunk_w] += w_2d

        tail = ext_w - W
        if tail > 0:
            acc[:, :tail, :] += acc[:, W:W + tail, :]
            wsum[:, :tail] += wsum[:, W:W + tail]

        result = acc[:, :W, :] / wsum[:, :W, None]
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def edit(self, image_path, prompt, mode="general", output_path=None):
        """Edit an image with a freeform prompt.

        mode:
          - "general": Lightning LoRA only (any prompt).
          - "remove_objects": Lightning + Object-Remover LoRAs stacked.

        For ~2:1 panoramas, the edit is tiled into 4 overlapping square chunks
        (with seam-wrap on the last chunk) and blended back together. For
        non-panoramic aspect ratios, the model runs once on the whole image.
        """
        if mode == "remove_objects":
            self.pipe.set_adapters(["speed", "remove"], adapter_weights=[1.0, 1.0])
        else:
            self.pipe.set_adapters("speed")

        input_image = load_image(image_path)
        aspect = input_image.width / input_image.height
        if aspect >= PANO_ASPECT_THRESHOLD:
            print(f"Editing pano {image_path} (mode={mode}, chunked)...")
            image = self._chunked_pano_edit(input_image, prompt)
        else:
            print(f"Editing image {image_path} (mode={mode})...")
            image = self._run_pipe(input_image, prompt)

        if output_path:
            image.save(output_path)
            print(f"Saved edited image to {output_path}")
        return image

    def clean(self, image_path, prompt=REMOVE_DEFAULT_PROMPT, output_path=None):
        return self.edit(image_path, prompt, mode="remove_objects", output_path=output_path)
