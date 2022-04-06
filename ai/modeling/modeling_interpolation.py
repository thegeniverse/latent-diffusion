import json
import gc
import math
from typing import *

import logging
import torch
import torchvision
from PIL import Image

from ai.modeling import model_factory
from ai.image_utils import generate_video_from_img_list
from ai.modeling.config import MODEL_NAME, OPTIMIZABLE_MODELS
from ai.services.storage import S3

FPS = 16


class InterpolationDataProcessor:
    def __init__(
        self,
        model_name: str,
        logger=logging,
    ):
        self.model_name = model_name
        self.logger = logger

    def interpolator(
        self,
        data_dict: Dict[str, Any],
    ):
        print("Data received")
        print(data_dict)
        
        self.logger.info(f"Processing interpolation data...")

        user_id = data_dict["userId"]
        canvas_element_list = json.loads(
            data_dict["canvasElements"])
        

        latents_list = []
        duration_list = []

        canvas_element_list.sort(
            key=lambda x: int(x['order']),
            reverse=False,
        )
        

        # if resolution is not None:
        #     latent_resolution = [
        #         max(resolution[0] // 16, 1),
        #         max(resolution[0] // 16, 1)
        #     ]
        
            
        latent_resolution = None

        for interpolation_element in canvas_element_list:
            img_id = interpolation_element.get("imgId")
            latents_id = interpolation_element.get("latentsId")
            bucket_name = interpolation_element.get("bucketName")
            region_name = interpolation_element.get("regionName")
        
            storage = S3(
                bucket_name=bucket_name,
                region_name=region_name,
            )

            interpolation_latent = storage.download_tensor(tensor_id=latents_id, )
            interpolation_latent = torch.tensor(interpolation_latent).clone().cuda()

            if latent_resolution is None:
                latent_resolution = interpolation_latent.shape[2::]

            else:
                interpolation_latent = torch.nn.functional.interpolate(
                    interpolation_latent,
                    latent_resolution,
                    mode="bilinear",
                )

            latents_list.append(interpolation_latent, )

            duration = interpolation_element["duration"]
            duration_list.append(duration)

        processed_data_dict = dict(
            user_id=user_id,
            latents_list=latents_list,
            duration_list=duration_list,
        )

        return processed_data_dict

    def __call__(
        self,
        data_dict: Dict[str, Any],
    ):
        processed_data_dict = self.interpolator(data_dict, )
        return processed_data_dict


class ImageInterpolator:
    def __init__(
        self,
        model_name: str = None,
        logger=logging,
    ):
        model_name = MODEL_NAME if model_name is None else model_name

        self.logger = logger

        self.hook_dict = {
            "results": None,
            "step": None,
            "error": None,
        }
        self.is_optimizable = model_name in OPTIMIZABLE_MODELS

        self.model = model_factory.load(model_name, )
        self.data_processor = InterpolationDataProcessor(model_name, )

    def register_results_hook(
        self,
        cb: Callable,
    ):
        self.hook_dict["results"] = cb

    def register_step_hook(
        self,
        cb: Callable,
    ):
        self.hook_dict["step"] = cb

    def register_error_hook(
        self,
        cb: Callable,
    ):
        self.hook_dict["error"] = cb

    def interpolate(
        self,
        user_id: str,
        latents_list,
        duration_list,
        device: str = "cuda",
        interpolate_mode="sin",  # sin | lineal
        emit=True,
        loop=True,
        **kwargs,
    ):
        gen_img_list = []

        for idx, (latents,
                  duration) in enumerate(zip(latents_list, duration_list)):

            if idx + 1 == len(latents_list) and not loop:
                break

            latents = latents.to(device)

            num_iterations = int(duration * FPS)
            init_latents = latents
            target_latents = latents_list[(idx + 1) % len(latents_list)]
            target_latents = target_latents.to(device)

            for step in range(num_iterations):
                if interpolate_mode == "sin":
                    weight = math.sin(1.5708 * step / num_iterations)**2
                else:
                    weight = step / num_iterations

                latents = weight * target_latents + (1 - weight) * init_latents

                tensor_img = self.model.get_img_from_latents(latents, )

                pil_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    tensor_img[0])

                gen_img_list.append(pil_img)

                if emit:
                    event_name = "gen-process"
                    gen_step = int(idx) * num_iterations + int(step)
                    gen_num_iterations = num_iterations * len(latents_list)
                    data_dict = {
                        "event": event_name,
                        "msg": f"Step {gen_step} / {gen_num_iterations}",
                        "userId": user_id,
                    }

                    results_cb = self.hook_dict["step"]
                    if results_cb is not None:
                        results_cb(data_dict, )

                torch.cuda.empty_cache()
                gc.collect()

        if emit:
            video_path, thumbnail_path = generate_video_from_img_list(
                user_id=user_id,
                img_list=gen_img_list,
                prefix="interpolation-",
            )

            thumbnail = Image.open(thumbnail_path, )

            event_name = "interpolation-results"
            data_dict = {
                "step": step,
                "userId": user_id,
                "event": event_name,
                "video": video_path,
                "thumbnail": thumbnail,
            }

            results_cb = self.hook_dict["results"]
            if results_cb is not None:
                results_cb(data_dict, )

            return data_dict

        return gen_img_list

    def start_process(
        self,
        data_dict: Dict[str, Any],
    ):
        try:
            processed_data_dict = self.data_processor(data_dict, )
            result = self.interpolate(**processed_data_dict, )
        
        except Exception as e:
            error_cb = self.hook_dict.get("error")
            if error_cb is not None:
                error_data_dict = dict(
                    exception = e,
                )
                error_cb(error_data_dict, )

        return result
