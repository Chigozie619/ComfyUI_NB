import os
import io
import numpy as np
import torch
from PIL import Image

try:
    import replicate
except ImportError:
    raise ImportError("replicate package not found. Run: pip install replicate")


def _tensor_to_bytesio(tensor):
    """ComfyUI IMAGE tensor [B, H, W, C] float32 0-1 → PNG BytesIO."""
    arr = (tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _bytes_to_tensor(image_bytes):
    """Raw image bytes → ComfyUI IMAGE tensor [1, H, W, C] float32 0-1."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class NanaBananaProNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["9:16", "16:9", "1:1", "4:3", "3:4"],),
                "resolution": (["4K", "2K", "1K"],),
            },
            "optional": {
                "image_1":  ("IMAGE",),
                "image_2":  ("IMAGE",),
                "image_3":  ("IMAGE",),
                "image_4":  ("IMAGE",),
                "image_5":  ("IMAGE",),
                "image_6":  ("IMAGE",),
                "image_7":  ("IMAGE",),
                "image_8":  ("IMAGE",),
                "image_9":  ("IMAGE",),
                "image_10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "run"
    CATEGORY = "Bloomstudios/API"

    def run(
        self,
        prompt,
        aspect_ratio,
        resolution,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
    ):
        api_key = os.environ.get("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable is not set."
            )

        raw_images = [
            image_1, image_2, image_3, image_4, image_5,
            image_6, image_7, image_8, image_9, image_10,
        ]
        image_input = [
            _tensor_to_bytesio(img) for img in raw_images if img is not None
        ]

        api_input = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
        if image_input:
            api_input["image_input"] = image_input

        output = replicate.run("google/nano-banana-pro", input=api_input)
        image_bytes = output.read()

        return (_bytes_to_tensor(image_bytes), prompt)


NODE_CLASS_MAPPINGS = {
    "NanaBananaPro": NanaBananaProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanaBananaPro": "Nana Banana Pro",
}
