import logging
from typing import *

from ai.modeling import ImageGenerator, ImageInterpolator, ImageZoomer


class ModelingFactory:
    def __init__(
        self,
        logger=logging,
    ):
        self.logger = logger

        self.image_generator = ImageGenerator()
        self.image_interpolator = ImageInterpolator()
        self.image_zoomer = ImageZoomer()

    def load(
        self,
        modeling_name,
    ):
        if modeling_name == "generate":
            return self.image_generator

        elif modeling_name == 'interpolate':
            return self.image_interpolator

        elif modeling_name == 'zoom':
            return self.image_zoomer


modeling_factory = ModelingFactory()