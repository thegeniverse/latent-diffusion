import torch
from ai.modeling import model_factory


class ImageUpscaler:
    def __init__(
        self,
        model_name: str = "esrgan",
    ):
        self.model = model_factory.load(model_name, )

    def upscale(
        self,
        img_tensor: torch.Tensor,
    ):
        upscaled_img = self.model(img_tensor, )
        return upscaled_img