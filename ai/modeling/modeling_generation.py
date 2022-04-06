import json
import gc
import logging
import time
from typing import *

import torch
import torchvision
from PIL import Image

from ai.modeling import model_factory
from ai.image_utils import pil_to_base64
from ai.modeling.config import MODEL_NAME, OPTIMIZABLE_MODELS
from ai.services.storage import S3


class GenerationDataProcessor:
    def __init__(
        self,
        model_name: str,
        logger=logging,
    ):
        self.model_name = model_name
        self.logger = logger

    def latent_optimizer(
        self,
        data_dict: Dict[str, Any],
    ):
        user_id = data_dict["userId"]

        num_steps = data_dict.get("numSteps")
        learning_rate = data_dict.get("learningRate")
        noise_factor = data_dict.get("noiseFactor")
        resolution = data_dict.get("resolution")
        num_crops = data_dict.get("numCrops")
        loss_type = data_dict.get("lossType")
        num_accum_steps = data_dict.get("numAccumSteps")

        prompt_list = []
        prompt_weight_list = []

        init_img = None
        init_latent = None

        canvas_elements = json.loads(data_dict["canvasElements"])
        for canvas_element in canvas_elements:
            element_type = canvas_element["type"]

            if element_type == "text":
                prompt_list.append(canvas_element["value"])

                # NOTE:  size 0 makes the weight to be ~1 and size 2 to be ~2
                weight = ((canvas_element["weight"] + 2) / 5 + 0.01) * 2.5

                influence = canvas_element["influence"]
                if influence == "-":
                    weight = -weight

                prompt_weight_list.append(weight)

            elif element_type == "img":
                img_id = canvas_element.get("imgId")
                latents_id = canvas_element.get("latentsId")
                bucket_name = canvas_element.get("bucketName")
                region_name = canvas_element.get("regionName")

                if any([
                        item is None or item == 0
                        for item in [img_id, bucket_name, region_name]
                ]):
                    continue

                storage = S3(
                    bucket_name=bucket_name,
                    region_name=region_name,
                )

                init_img = storage.download_pil(img_id, )

                if list(init_img.size) == resolution:
                    init_latent = storage.download_tensor(latents_id, )
                    init_img = None

                else:
                    init_img = storage.download_pil(img_id, )

                    img_width, img_height = init_img.size
                    scale_factor = max(resolution) / max(img_width, img_height)
                    target_size = (int(img_width * scale_factor),
                                   int(img_height * scale_factor))
                    init_img = init_img.resize(
                        target_size,
                        resample=Image.BILINEAR,
                    )

                    resolution = target_size

        prompt_list = [prompt.replace("\n", "") for prompt in prompt_list]
        self.logger.debug("Prompt list: ", " | ".join(prompt_list))

        processed_data_dict = dict(
            user_id=user_id,
            prompt_list=prompt_list,
            prompt_weight_list=prompt_weight_list,
            num_steps=num_steps,
            resolution=resolution,
            init_img=init_img,
            loss_type=loss_type,
            learning_rate=learning_rate,
            num_crops=num_crops,
            noise_factor=noise_factor,
            init_latents=init_latent,
            num_accum_steps=num_accum_steps,
        )

        processed_data_dict = {
            key: value
            for key, value in processed_data_dict.items() if value is not None
        }

        return processed_data_dict

    def automatic_latent_optimizer(
        self,
        data_dict: Dict[str, Any],
    ):
        pass

    def pipelined_latent_optimizer(
        self,
        data_dict: Dict[str, Any],
    ):
        pass

    def single_latent_optimizer(
        self,
        data_dict: Dict[str, Any],
    ):
        num_res = data_dict["numRes"]

    def __call__(
        self,
        data_dict: Dict[str, Any],
    ):
        if self.model_name in OPTIMIZABLE_MODELS:
            if "automatic" in data_dict.keys():
                processed_data_dict = self.automatic_latent_optimizer(
                    data_dict, )

            elif "pipeline" in data_dict.keys():
                processed_data_dict = self.pipelined_latent_optimizer(
                    data_dict, )

            else:
                processed_data_dict = self.latent_optimizer(data_dict, )

        else:
            processed_data_dict = self.single_latent_optimizer(data_dict, )

        return processed_data_dict


class ImageGenerator:
    def __init__(
        self,
        model_name: str = None,
        logger=logging,
    ):
        self.logger = logger

        model_name = MODEL_NAME if model_name is None else model_name

        self.stop = False
        self.hook_dict = {
            "error": None,
            "step": None,
            "results": None,
        }
        self.is_optimizable = model_name in OPTIMIZABLE_MODELS

        self.model = model_factory.load(model_name, )
        self.data_processor = GenerationDataProcessor(model_name, )

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

    def stop_process(self, ):
        self.stop = True

    def optimize_latents(
        self,
        user_id: str,
        prompt_list: List[List[str]],
        prompt_weight_list: List[List[float]],
        num_steps: int,
        resolution: Tuple[int],
        learning_rate: float = 0.5,
        init_img: Union[Image.Image, torch.Tensor] = None,
        init_latents: torch.Tensor = None,
        loss_type="cosine_similarity",
        num_crops: int = 16,
        noise_factor: float = 0.11,
        num_accum_steps: int = 8,
        init_step: int = 0,
        upscale: bool = False,
        num_total_iterations: int = None,
        device: str = "cuda",
        **kwargs,
    ):
        try:
            batch_size = 1

            max_time = 60
            time_start = time.time()

            if num_total_iterations is None:
                num_total_iterations = num_steps

            self.logger.warning("usinc vicc's hardcoded accum steps")
            if sum(resolution) > 400 + 400:
                num_accum_steps = 4
            elif sum(resolution) > 256 + 256:
                if num_crops > 64:
                    num_accum_steps = 4
                else:
                    num_accum_steps = 2
            else:
                num_accum_steps = 1

            num_crops = max(1, int(num_crops / num_accum_steps))
            self.logger.debug(f"Using {num_crops} augmentations")
            self.logger.debug(f"Using {num_accum_steps} accum steps")
            self.logger.debug(
                f"Effective num crops of {num_accum_steps * num_crops}")

            assert loss_type in self.model.supported_loss_types, f"ERROR! Loss type " \
                f"{loss_type} not recognized. " f"Only " \
                f"{' or '.join(self.model.supported_loss_types)} supported."

            if init_latents is not None:
                latents = init_latents.to(device, )
                init_img = self.model.get_img_from_latents(latents, )
                init_img_size = init_img.shape[2::]

                scale = (max(resolution) // 16 * 16) / max(init_img_size)
                if scale != 1:
                    img_resolution = [
                        int(init_img_size[0] * scale),
                        int(init_img_size[1] * scale)
                    ]

                    init_img = torch.nn.functional.interpolate(
                        init_img,
                        img_resolution,
                        mode="bilinear",
                    )
                    latents = self.model.get_latents_from_img(init_img, )

            elif init_img is not None:
                latents = self.model.get_latents_from_img(init_img, )

            else:
                latents = self.model.get_random_latents(
                    batch_size=batch_size,
                    target_img_width=resolution[0],
                    target_img_height=resolution[1],
                )

            latents = latents.to(device)
            latents = torch.nn.Parameter(latents)

            optimizer = torch.optim.AdamW(
                params=[latents],
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.1,
            )

            for step in range(init_step, init_step + num_steps):
                optimizer.zero_grad()

                for _num_accum in range(num_accum_steps):
                    loss = 0

                    img_rec = self.model.get_img_from_latents(latents, )

                    x_rec_stacked = self.model.augment(
                        img_rec,
                        num_crops=num_crops,
                        noise_factor=noise_factor,
                    )

                    img_logits_list = self.model.get_clip_img_encodings(
                        x_rec_stacked, )

                    for prompt, prompt_weight in zip(prompt_list,
                                                     prompt_weight_list):
                        text_logits_list = self.model.get_clip_text_encodings(
                            prompt, )

                        for img_logits, text_logits in zip(
                                img_logits_list, text_logits_list):
                            text_logits = text_logits.clone().detach()
                            if loss_type == 'cosine_similarity':
                                clip_loss = -10 * torch.cosine_similarity(
                                    text_logits, img_logits).mean()

                            if loss_type == "spherical_distance":
                                clip_loss = (text_logits - img_logits).norm(
                                    dim=-1).div(2).arcsin().pow(2).mul(
                                        2).mean()

                            loss += prompt_weight * clip_loss

                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if step == init_step + num_steps - 1 and upscale:
                    img_rec = self.upscale(img_rec, )

                gen_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    img_rec[0], )

                max_compressed_resolution = 400

                if max(gen_img.size) > max_compressed_resolution:
                    compressed_ratio = max(
                        gen_img.size) / max_compressed_resolution

                    compressed_resolution = (
                        int(gen_img.size[0] / compressed_ratio),
                        int(gen_img.size[1] / compressed_ratio),
                    )

                    compressed_gen_img = gen_img.resize(compressed_resolution, )
                
                else:
                    compressed_gen_img = gen_img

                finished = step == (init_step + num_steps - 1)
                
                if time.time() - time_start > max_time:
                    self.stop = True
                    finished = True

                event_name = "generation-results"
                data_dict = {
                    "event": event_name,
                    "userId": user_id,
                    "step": step,
                    "img": gen_img,
                    "thumbnail": compressed_gen_img,
                    "latents": latents.detach().clone().cpu(),
                    "numSteps": num_total_iterations,
                    "prompt": "-".join(['_'.join(p for p in prompt.split(" ")) for prompt in prompt_list]),
                    "finished": finished,
                }

                results_cb = self.hook_dict["results"]
                if results_cb is not None:
                    results_cb(data_dict)


                if self.stop:
                    if upscale:
                        img_rec = self.upscale(img_rec, )

                    self.stop = False

                    return gen_img, latents

                torch.cuda.empty_cache()
                gc.collect()

            if upscale:
                img_rec = self.upscale(img_rec, )
                gen_img = torchvision.transforms.ToPILImage(mode='RGB')(
                    img_rec[0], )

            gen_latents = latents.detach().clone().cpu()
            
            del latents
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()

            return gen_img, gen_latents

        except Exception as e:
            error_cb = self.hook_dict.get("error")
            if error_cb is not None:
                error_data_dict = dict(
                    userId=user_id,
                    exception=e,
                )
                error_cb(error_data_dict)

    def one_step_optimization(
        self,
        user_id: str,
        prompt_list,
        num_res: int = 1,
        mode: str = "clip_guided",
        **kwargs,
    ):
        prompt = " ".join(prompt_list)
        upsample_temp = 0.997,
        guidance_scale = 5.

        gen_img_list = self.model.generate_from_prompt(
            prompt=prompt,
            mode=mode,
            num_results=num_res,
            guidance_scale=guidance_scale,
            upsample_temp=upsample_temp,
        )

        event_name = "gen-results"
        gen_img_base64 = [pil_to_base64(gen_img) for gen_img in gen_img_list]

        data_dict = {
            "event": event_name,
            "userId": user_id,
            "msg": f"One step optimization...",
            "genImg": gen_img_base64,
            "numIterations": 1,
            "step": 0,
            "loss": 0,
            "numIterRate": 1,
            "latents": None,
            "is_pro": True,
        }

        cb_optimization = self.hook_dict["results"]
        if cb_optimization is not None:
            cb_optimization(data_dict, )

        return

    def free_optimization(
        self,
        user_id: str,
        generation_dict: Dict[str, Any],
    ):
        try:
            auto_params_list = [
                {
                    "resolution": (128, 128),
                    "learning_rate": 0.6,
                    "num_steps": 10,
                    "num_crops": 64,
                    "num_accum_steps": 1,
                    "upscale": False,
                    "send_freq": 1,
                },
                {
                    "resolution": (128, 128),
                    "learning_rate": 0.2,
                    "num_steps": 10,
                    "num_crops": 64,
                    "num_accum_steps": 1,
                    "upscale": False,
                    "send_freq": 1,
                },
                {
                    "resolution": (256, 256),
                    "learning_rate": 0.3,
                    "num_steps": 10,
                    "num_crops": 32,
                    "num_accum_steps": 1,
                    "upscale": False,
                    "send_freq": 1,
                },
                {
                    "resolution": (256, 256),
                    "learning_rate": 0.2,
                    "num_steps": 10,
                    "num_crops": 32,
                    "num_accum_steps": 1,
                    "upscale": True,
                    "send_freq": 1,
                },
            ]

            num_total_iterations = sum(
                [params["num_steps"] for params in auto_params_list])

            init_step = 0
            for params_idx, auto_params in enumerate(auto_params_list):
                generation_dict.update(auto_params, )

                gen_img, latents = self.optimize_latents(
                    user_id=user_id,
                    num_total_iterations=num_total_iterations,
                    init_step=init_step,
                    is_pro=False,
                    **generation_dict,
                )

                if auto_params["upscale"]:
                    generation_dict["init_latents"] = None

                    gen_w, gen_h = gen_img.size
                    generation_dict["init_img"] = gen_img.resize(
                        (int(gen_w / 4), int(gen_h / 4)))

                else:
                    generation_dict["init_latents"] = latents
                    generation_dict["init_img"] = None

                init_step += auto_params["num_steps"]

        except Exception as e:
            self.logger.error("Error in automatic generation...")

            error_cb = self.hook_dict["error"]
            if error_cb is not None:
                error_dict = {
                    "userId": user_id,
                    "exception": e,
                }
                error_cb(error_dict, )

    def generate(
        self,
        data_dict: Dict[str, Any],
    ):
        processed_data_dict = self.data_processor(data_dict, )
        if self.is_optimizable:
            result = self.optimize_latents(**processed_data_dict, )

        else:
            pass

        return result

    def start_process(
        self,
        data_dict: Dict[str, Any],
    ):
        try:
            result = self.generate(data_dict, )

        except Exception as e:
            error_cb = self.hook_dict.get("error")
            if error_cb is not None:
                error_data_dict = dict(
                    exception = e,
                )
                error_cb(error_data_dict, )
                

        return result
