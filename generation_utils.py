import gc
import os
from typing import *

import torch
import torchvision
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class LatentDiffusionModel:
    def __init__(
        self,
        config_path: str = "configs/latent-diffusion/txt2img-1p4B-eval.yaml",
        model_ckpt_path: str = "models/ldm/text2img-large/model.ckpt",
        outdir: str = "output/txt2img-samples",
        device: str = "cuda:0",
    ):
        self.device = torch.device(device, )

        os.makedirs(
            outdir,
            exist_ok=True,
        )
        self.outpath = outdir

        self.sample_path = os.path.join(
            self.outpath,
            "samples",
        )
        os.makedirs(
            self.sample_path,
            exist_ok=True,
        )

        config = OmegaConf.load(config_path, )

        self.model = self.load_model_from_config(
            config,
            model_ckpt_path,
        )

    def load_model_from_config(
        self,
        config: OmegaConf,
        ckpt_path: str,
        verbose: bool = False,
        mixed_precision: bool = False,
    ):
        print(f"Loading model from {ckpt_path}")

        pl_sd = torch.load(
            ckpt_path,
            map_location="cpu",
        )

        sd = pl_sd["state_dict"]

        model = instantiate_from_config(config.model, )
        missing_keys, unexpected_keys = model.load_state_dict(
            sd,
            strict=False,
        )

        if len(missing_keys) > 0 and verbose:
            print("missing keys:")
            print(missing_keys)

        if len(unexpected_keys) > 0 and verbose:
            print("unexpected keys:")
            print(unexpected_keys)

        if mixed_precision:
            model.half()

        model.to(self.device)

        model.eval()

        return model

    def generate_from_prompt(
        self,
        prompt_list: List[str],
        ddim_steps: int = 100,
        ddim_eta: float = 0.0,
        plms: bool = True,
        n_iter: int = 1,
        img_height: int = 256,
        img_width: int = 256,
        n_samples: int = 4,
        temperature: float = 1.0,
        scale: float = 10.0,
        negative_scale: float = 10.0,
        num_grid_rows: int = 2,
        save_result: bool = False,
        save_intermediates: bool = False,
        seed: int = None,
    ):
        torch.cuda.empty_cache()
        gc.collect()

        if seed is not None:
            torch.manual_seed(seed, )

        if isinstance(prompt_list, str):
            prompt_list = [
                prompt_list,
            ]

        num_saved_imgs = len(os.listdir(self.sample_path, ))

        if plms:
            print("Using plms")

            if ddim_eta != 0:
                print("Warning! Using plms samples with a ddim_eta != 0")

            sampler = PLMSSampler(self.model, )

        else:
            sampler = DDIMSampler(self.model, )

        shape = [4, img_height // 8, img_width // 8]

        batch_size = n_samples * len(prompt_list)

        all_samples = list()
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            with self.model.ema_scope():
                uc = None

                if scale != 1.0:
                    uc = self.model.get_learned_conditioning(
                        batch_size * [""], )

                for _ in trange(n_iter, desc="Sampling"):
                    # init_noise = torch.randn(
                    #     [batch_size, *shape],
                    #     device=self.device,
                    # )

                    x = torchvision.transforms.ToTensor()(
                        Image.open("./output/shark.jpg").resize(
                            (256, 256))).to(self.device)[None, :] * 2 - 1
                    encoder_posterior = self.model.encode_first_stage(x)
                    init_noise = self.model.get_first_stage_encoding(
                        encoder_posterior).detach()
                    # init_noise = None

                    c = self.model.get_learned_conditioning([
                        prompt for _ in range(n_samples)
                        for prompt in prompt_list
                    ], )

                    c_negative = self.model.get_learned_conditioning(
                        ["psychedelic image"], )
                    c_negative = torch.cat([c_negative] * batch_size)

                    c_negative = None

                    shape = [4, img_height // 8, img_width // 8]

                    samples_ddim, intermediates = sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        conditioning_negative=c_negative,
                        batch_size=batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        negative_unconditional_guidance_scale=negative_scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        temperature=temperature,
                        x_T=init_noise,
                    )

                    if save_intermediates:
                        for inter_idx, intermediate in enumerate(
                                intermediates["pred_x0"]):
                            x_samples_ddim = self.model.decode_first_stage(
                                intermediate, )

                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0,
                                min=0.0,
                                max=1.0,
                            )

                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(
                                    x_sample.detach().cpu().numpy(),
                                    'c h w -> h w c',
                                )

                                Image.fromarray(
                                    x_sample.astype(np.uint8)
                                ).save(
                                    os.path.join(
                                        self.sample_path,
                                        f"{num_saved_imgs:04}_{inter_idx}.png")
                                )

                        # for inter_idx, intermediate in enumerate(
                        #         intermediates["x_inter"]):
                        #     x_samples_ddim = self.model.decode_first_stage(
                        #         intermediate, )

                        #     x_samples_ddim = torch.clamp(
                        #         (x_samples_ddim + 1.0) / 2.0,
                        #         min=0.0,
                        #         max=1.0,
                        #     )

                        #     for x_sample in x_samples_ddim:
                        #         x_sample = 255. * rearrange(
                        #             x_sample.detach().cpu().numpy(),
                        #             'c h w -> h w c',
                        #         )

                        #         Image.fromarray(
                        #             x_sample.astype(np.uint8)
                        #         ).save(
                        #             os.path.join(
                        #                 self.sample_path,
                        #                 f"{num_saved_imgs:04}_{inter_idx}.png")
                        #         )

                    x_samples_ddim = self.model.decode_first_stage(
                        samples_ddim, )

                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0,
                        min=0.0,
                        max=1.0,
                    )

                    if save_result:
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(
                                x_sample.detach().cpu().numpy(),
                                'c h w -> h w c',
                            )

                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(self.sample_path,
                                             f"{num_saved_imgs:04}.png"))

                        num_saved_imgs += 1

                    all_samples.append(x_samples_ddim, )

        grid = torch.stack(all_samples, 0)
        grid = rearrange(
            grid,
            'n b c h w -> (n b) c h w',
        )

        if num_grid_rows is None:
            num_grid_rows = 2

        grid = torchvision.utils.make_grid(
            grid,
            nrow=num_grid_rows,
        )

        grid = 255. * rearrange(grid, 'c h w -> h w c').detach().cpu().numpy()
        pil_img = Image.fromarray(grid.astype(np.uint8))

        if save_result:
            pil_img.save(
                os.path.join(
                    self.outpath,
                    f'{num_saved_imgs}-{prompt_list[0].replace(" ", "-")}.png')
            )

            print(
                f"Your samples are ready and waiting four you here: \n{self.outpath} \nEnjoy."
            )

        return pil_img


if __name__ == "__main__":
    prompt_list = [
        "an illustration of a baby radish in tutu walking a dog",
        # "3D concept character. Robot shark rendered with unity. Robot shark trending on artstation. A character in 3D of a robot shark",
        # "artstation artwork, psychedelic painting of a cat",
        #"artstation artwork, psychedelic painting of a cat",
        #"artstation artwork, psychedelic painting of a cat",
        #"artstation artwork, psychedelic painting of a gorilla",
        #"artstation artwork, psychedelic painting of a elephant",
        #"artstation artwork, psychedelic painting of a lion",
    ]
    n_samples = 1

    model = LatentDiffusionModel()

    img = model.generate_from_prompt(
        prompt_list,
        plms=False,
        ddim_steps=200,
        n_samples=n_samples,
        n_iter=1,
        num_grid_rows=2,
        save_result=True,
        save_intermediates=True,
        seed=666,
        scale=13.,
        negative_scale=0.,
    )
