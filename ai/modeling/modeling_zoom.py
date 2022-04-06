import json
import gc
import math
from typing import *

import logging
import torch
import torchvision

from ai.modeling import model_factory
from ai.image_utils import generate_video_from_img_list, base64_to_PIL
from ai.modeling.config import MODEL_NAME, OPTIMIZABLE_MODELS
from ai.modeling import ImageInterpolator

FPS = 16


class ImageZoomer:
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

        self.model = model_factory.load(model_name=model_name, )
        self.image_interpolator = ImageInterpolator(model_name=model_name)

        self.stop = False

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

    def preprocess(
        self,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.logger.info(f"Processing interpolation data...")
        model_name = data_dict["modelName"]
        user_id = data_dict["userID"]
        num_crops = data_dict["numCrops"]

        data_list = json.loads(data_dict["data"])

        prompt_list = []
        prompt_weight_list = []
        for data in data_list:
            data_type = data["type"]
            if data_type == "text":
                prompt_list.append(data["content"])

                weight = ((data["size"] + 2) / 5 + 0.01) * 2.5

                mode = data["mode"]
                if mode == "-":
                    weight = -weight

                prompt_weight_list.append(weight)

            elif data_type == "img":
                latents = torch.tensor(data["embed"][0])
                duration = data["duration"]

        self.logger.info(f"{user_id} PROMPT LIST --> {prompt_list}")
        self.logger.info(f"{user_id} LATENTS SHAPE --> {latents.shape}")
        self.logger.info(f"{user_id} DURATION --> {duration}")

        lr = data_dict["zoomLR"]
        num_zoom_interp_steps = data_dict["zoomOptInterpSteps"]
        num_zoom_train_steps = data_dict["zoomOptTrainSteps"]
        zoom_offset = data_dict["zoomOptOffset"]

        zoom_dict = dict(
            mode="zoom",
            user_id=user_id,
            prompt_list=prompt_list,
            prompt_weight_list=prompt_weight_list,
            model_name=model_name,
            init_latents=latents,
            duration=duration,
            lr=lr,
            num_zoom_interp_steps=num_zoom_interp_steps,
            num_zoom_train_steps=num_zoom_train_steps,
            zoom_offset=zoom_offset,
            num_crops=num_crops,
        )

        return zoom_dict

    def zoom(
        self,
        user_id: str,
        model_name: str,
        prompt_list: List[str],
        prompt_weight_list: List[str],
        init_latents: torch.Tensor,
        device: str = "cuda",
        duration: float = 5.,
        lr=0.0008,
        num_zoom_interp_steps=2,
        num_zoom_train_steps=8,
        zoom_offset=4,
        num_crops=64,
        loss_type="cosine_similarity",
        **kwargs,
    ):
        num_iterations = duration * FPS

        latents = init_latents.detach().clone().to(device)
        latents.requires_grad = True
        latents = torch.nn.Parameter(latents)

        optimizer = torch.optim.AdamW(
            params=[latents],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        gen_img_list = []
        for step in range(num_iterations):
            if self.stop:
                self.stop = False
                print("Stopped")
                break

            with torch.no_grad():
                tensor_img = self.model.get_img_from_latents(latents, )

                zoom_tensor_img = tensor_img[:, :, zoom_offset:-zoom_offset,
                                             zoom_offset:-zoom_offset, ]

                img_height, img_width = tensor_img.shape[-2::]
                zoom_tensor_img = torch.nn.functional.interpolate(
                    zoom_tensor_img,
                    (img_height, img_width),
                    mode="bilinear",
                )

                zoom_latents = self.model.get_latents_from_img(
                    torchvision.transforms.ToPILImage(mode="RGB")(
                        zoom_tensor_img[0]), )

                latents.data = zoom_latents.data

            for zoom_train_step in range(num_zoom_train_steps):
                loss = 0

                tensor_img = self.model.get_img_from_latents(latents, )
                self.logger.info(f"TENSOR IMG SHAPE {tensor_img.shape}")
                tensor_img_stack = self.model.augment(
                    tensor_img,
                    num_crops=num_crops,
                    #tensor_img.shape[1],
                    #tensor_img.shape[2],
                )

                img_logits_list = self.model.get_clip_img_encodings(
                    tensor_img_stack, )

                for prompt, prompt_weight in zip(prompt_list,
                                                 prompt_weight_list):
                    text_logits_list = self.model.get_clip_text_encodings(
                        prompt, )

                    for img_logits, text_logits in zip(img_logits_list,
                                                       text_logits_list):
                        if loss_type == 'cosine_similarity':
                            clip_loss = -10 * torch.cosine_similarity(
                                text_logits, img_logits).mean()

                        if loss_type == "spherical_distance":
                            clip_loss = (text_logits - img_logits).norm(
                                dim=-1).div(2).arcsin().pow(2).mul(2).mean()

                        loss += prompt_weight * clip_loss

                event_name = "gen-process"
                data_dict = {
                    "event": event_name,
                    "msg": f"Step {step} / {num_iterations}",
                    "userID": user_id,
                    "step": step,
                }

                step_cb = self.hook_dict["step"]
                if step_cb is not None:
                    step_cb(data_dict, )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()
                gc.collect()

            with torch.no_grad():
                interpolate_latents_list = [init_latents, latents]
                interpolate_duration_list = [num_zoom_interp_steps / FPS] * 2

                interpolate_img_list = self.image_interpolator.interpolate(
                    user_id=user_id,
                    model_name=model_name,
                    latents_list=interpolate_latents_list,
                    duration_list=interpolate_duration_list,
                    interpolate_mode="lineal",
                    emit=False,
                    loop=False,
                )

                gen_img_list += interpolate_img_list

            init_latents = latents.detach().clone()

            torch.cuda.empty_cache()
            gc.collect()

        video_url, thumbnail_url = generate_video_from_img_list(
            user_id=user_id,
            img_list=gen_img_list,
            prefix="zoom-",
        )

        event_name = "zoom-res"
        data_dict = {
            "event": event_name,
            "success": True,
            "videoURL": video_url,
            "thumbnailURL": thumbnail_url,
            "user_id": user_id,
        }

        zoom_cb = self.hook_dict["results"]
        if zoom_cb is not None:
            zoom_cb(data_dict, )

        return gen_img_list

    def start_process(
        self,
        data_dict: Dict[str, Any],
    ):
        zoom_data_dict = self.preprocess(data_dict)

        result = self.zoom(**zoom_data_dict, )

        return result

    def stop_process(self, ):
        self.stop = True
