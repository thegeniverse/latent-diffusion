import gc
import logging
from typing import *
from collections import defaultdict

from upscaler.models import ESRGAN, ESRGANConfig

import torch
from geniverse.models import TamingDecoder, Aphantasia, Glide


class ModelFactory:
    def __init__(self, ):
        self.taming_decoder_dict = defaultdict(lambda: None)
        self.aphantasia_dict = defaultdict(lambda: None)
        self.glide_dict = defaultdict(lambda: None)
        self.upscaler_dict = defaultdict(lambda: None)

        self.clip_model_name_list = [
            "ViT-B/32",
            "ViT-B/16",
            # "RN50x16",
            #"RN50x4",
        ]

    def load(
        self,
        model_name: str,
        device: str = "cuda",
    ):
        if model_name != "Blender":
            if self.glide_dict[device] is not None:
                logging.info("CLEANING MEMORY AFTER BLENDER")
                torch.cuda.empty_cache()
                gc.collect()

        if "taming" in model_name or model_name == "Artisan":
            if "-" in model_name:
                model_type = model_name.split("-")[-1]
            else:
                model_type = "imagenet_16384"
                # model_type = "wikiart_16384"
                # model_type = "wikiart_7mil"

            logging.info("LOADING ARTISAN")
            if self.taming_decoder_dict[device] is None:
                logging.info("SETTTING ARTISAN")
                self.taming_decoder_dict[device] = TamingDecoder(
                    device=device,
                    clip_model_name_list=self.clip_model_name_list,
                    model_name=model_type,
                )
                self.taming_decoder_dict[device].share_memory()

            logging.info("ARTISAN READY")

            model = self.taming_decoder_dict[device]
            model = model.to(device)

        elif model_name == "aphantasia" or model_name == "Dreamer":
            if self.aphantasia_dict[device] is None:
                self.aphantasia_dict[device] = Aphantasia(
                    device=device,
                    clip_model_name_list=self.clip_model_name_list,
                )
                self.aphantasia_dict[device].share_memory()

            model = self.aphantasia_dict[device]
            model = model.to(device)

        elif model_name == "glide" or model_name == "Blender":
            logging.info("CLEANING MEMORY AFTER BLENDER - 2")

            torch.cuda.empty_cache()
            gc.collect()

            if self.glide_dict[device] is None:
                self.glide_dict[device] = Glide()
                self.glide_dict[device].share_memory()

            model = self.glide_dict[device]
            model = model.to(device)

        elif model_name == "upscaler":
            torch.cuda.empty_cache()
            gc.collect()

            if self.upscaler_dict[device] is None:
                model_name = "RealESRGAN_x4plus"
                tile = 256

                esrgan_config = ESRGANConfig(
                    model_name=model_name,
                    tile=tile,
                )

                self.upscaler_dict[device] = ESRGAN(esrgan_config, )

            model = self.upscaler_dict[device]

        return model


model_factory = ModelFactory()
