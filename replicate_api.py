import replicate
import os
import torch
from PIL import Image
from io import BytesIO
import numpy as np

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

class ReplicateBase:
    MODEL_ID = ""
    FUNCTION = "call"
    CATEGORY = "Replicate AI"

    def call(self, *args, **kwargs):
        kwargs['comfyui'] = True

        if kwargs.get("api_key_override"):
            os.environ["REPLICATE_API_TOKEN"] = kwargs.get("api_key_override")

        if REPLICATE_API_TOKEN is None:
            raise Exception(f"No Replicate API key set.\n\nUse your Replicate API key by:\n1. Setting the REPLICATE_API_TOKEN environment variable to your API key\n4. Passing the API key as an argument to the function with the key 'api_key_override'")

        output = replicate.run(
            self.MODEL_ID,
            input=kwargs
        )

        result_image = Image.open(BytesIO(output['image']))
        result_image = result_image.convert("RGBA")
        result_image = np.array(result_image).astype(np.float32) / 255.0
        result_image = torch.from_numpy(result_image)[None,]
        return (result_image,)

class NovakidStyler(ReplicateBase):
    MODEL_ID = "chesnokovivan/novakid-styler:6c9510194f7f74f3b65dbca0c431a422c6fae6a7309d807e149eb1afc4d03ef0"
    INPUT_SPEC = {
        "required": {
            "width": ("INT",),
            "height": ("INT",),
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "refine": ("STRING", {"default": "no_refiner"}),
            "scheduler": ("STRING", {"default": "K_EULER"}),
            "lora_scale": ("FLOAT", {"default": 0.6}),
            "num_outputs": ("INT", {"default": 1}),
            "guidance_scale": ("FLOAT", {"default": 7.5}),
            "apply_watermark": ("BOOL", {"default": True}),
            "high_noise_frac": ("FLOAT", {"default": 0.8}),
            "negative_prompt": ("STRING", {"default": ""}),
            "prompt_strength": ("FLOAT", {"default": 0.8}),
            "num_inference_steps": ("INT", {"default": 50}),
            "api_key_override": ("STRING", {"multiline": False}),
        }
    }