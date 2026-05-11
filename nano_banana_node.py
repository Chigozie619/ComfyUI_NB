import io
import os
import numpy as np
import torch
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai package not found. Run: pip install google-genai")


_KNOWN_RATIOS = [
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4",
    "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
]


def _tensor_to_pil(tensor):
    """ComfyUI IMAGE tensor [B, H, W, C] float32 0-1 → PIL Image."""
    arr = (tensor[0].numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(img):
    """PIL Image → ComfyUI IMAGE tensor [1, H, W, C] float32 0-1."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _closest_aspect_ratio(pil_image):
    """Return the closest known ratio string for a PIL image's dimensions."""
    w, h = pil_image.size
    target = w / h
    best = min(
        _KNOWN_RATIOS,
        key=lambda r: abs(int(r.split(":")[0]) / int(r.split(":")[1]) - target),
    )
    return best


class NanaBananaProNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["match_input_image", "21:9", "16:9", "9:16", "5:4", "4:5", "4:3", "3:4", "3:2", "2:3", "1:1"],),
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
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set."
            )

        raw_images = [
            image_1, image_2, image_3, image_4, image_5,
            image_6, image_7, image_8, image_9, image_10,
        ]
        pil_images = [_tensor_to_pil(img) for img in raw_images if img is not None]

        if aspect_ratio == "match_input_image":
            ar = _closest_aspect_ratio(pil_images[0]) if pil_images else "1:1"
        else:
            ar = aspect_ratio

        client = genai.Client(api_key=api_key)

        full_prompt = f"{prompt}\n\nOutput image aspect ratio: {ar}, resolution: {resolution}"
        contents = [full_prompt] + pil_images

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                pil_img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                return (_pil_to_tensor(pil_img), prompt)

        raise RuntimeError("Gemini API returned no image in the response.")


NODE_CLASS_MAPPINGS = {
    "NanaBananaPro": NanaBananaProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanaBananaPro": "Bloom Nana Banana Pro",
}
