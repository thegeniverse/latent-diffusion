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
from geniverse.modeling_utils import ImageGenerator

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

device = "cuda"
device = torch.device(device, )

outdir: str = "output/txt2img-samples"

os.makedirs(
    outdir,
    exist_ok=True,
)
outpath = outdir

sample_path = os.path.join(
    outpath,
    "samples",
)
os.makedirs(
    sample_path,
    exist_ok=True,
)


def load_model_from_config(
    config: OmegaConf,
    ckpt_path: str,
    verbose: bool = False,
    mixed_precision: bool = False,
    device: str = "cuda",
):
    print(f"Loading model from {ckpt_path}")

    pl_sd = torch.load(
        ckpt_path,
        map_location=device,
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

    # HACK: can make stuff more efficient
    if mixed_precision:
        model.half()

    # TODO: check if this is necessary
    #model.to(device, )

    # model.eval()

    return model


class LDMDecoder(ImageGenerator):
    def __init__(
        self,
        device: str = 'cuda',
        **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            **kwargs,
        )

        if device is not None:
            self.device = device

    def generate_from_prompt(self, *args, **kwargs):
        return super().generate_from_prompt(*args, **kwargs)

    def optimize_latent(
        self,
        prompt: str,
        shape: Tuple[int, ],
        lr: float = 0.08,
        num_steps: int = 16,
        num_augmentations: int = 32,
        target_img_width: int = 224,
        target_img_height: int = 224,
        loss_type: str = "cosine_similarity",
    ):
        z_logits = torch.randn(
            shape,
            device=device,
        )

        z_logits = torch.nn.Parameter(z_logits)
        z_logits.requires_grad = True

        optimizer = torch.optim.AdamW(
            params=[z_logits],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        for step_idx in range(num_steps, ):
            x_samples_ddim = model.decode_first_stage(z_logits[None, :], )

            x_samples_ddim = torch.clamp(
                (x_samples_ddim + 1.0) / 2.0,
                min=0.0,
                max=1.0,
            )

            x_rec_stacked = self.augment(
                x_samples_ddim.to(torch.float32),
                num_crops=num_augmentations,
                target_img_width=target_img_width,
                target_img_height=target_img_height,
            )

            clip_loss = 10 * self.compute_clip_loss(
                x_rec_stacked,
                prompt,
                loss_type,
            )

            clip_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torchvision.transforms.ToPILImage()(
                x_samples_ddim[0]).save(f"output/{step_idx}.png")

            print(f"Step {step_idx} done!")

        print(f"Optimization step")


config_path = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
config = OmegaConf.load(
    config_path,
)  # TODO: Optionally download from same location as ckpt_path and chnage this logic

model_ckpt_path = "models/ldm/text2img-large/model.ckpt"
model = load_model_from_config(
    config,
    model_ckpt_path,
    device=device,
)

model = model.to(device)

ldm_decoder = LDMDecoder()


def generate_from_prompt(
    prompt_list: List[str],
    ddim_steps: int = 100,
    ddim_eta: float = 0.0,
    plms: bool = False,
    n_iter: int = 1,
    H: int = 256,
    W: int = 256,
    n_samples: int = 4,
    temperature: float = 1.0,
    scale: float = 10.0,
    num_rows: int = 2,
    save_result: bool = False,
):
    if isinstance(prompt_list, str):
        prompt_list = [
            prompt_list,
        ]
    #torch.manual_seed(0)
    torch.cuda.empty_cache()
    gc.collect()

    #if prompt[0] == ".":
    #    prompt = prompt[1::]

    #prompt = prompt + ". Oil on canvas"

    base_count = len(os.listdir(sample_path))

    if plms:
        print("Using plms")
        if ddim_eta != 0:
            print("Warning! Using plms samples with a ddim_eta != 0")

        sampler = PLMSSampler(model, )

    else:
        sampler = DDIMSampler(model, )

    shape = [4, H // 8, W // 8]

    all_samples = list()
    with torch.no_grad():
        # with torch.cuda.amp.autocast():
        with model.ema_scope():
            uc = None

            if scale != 1.0:
                uc = model.get_learned_conditioning(
                    n_samples * len(prompt_list) * [""], )

            for _n in trange(n_iter, desc="Sampling"):
                c = model.get_learned_conditioning([
                    prompt for _ in range(n_samples) for prompt in prompt_list
                ], )
                shape = [4, H // 8, W // 8]

                with torch.enable_grad():
                    x0 = ldm_decoder.optimize_latent(
                        prompt=prompt_list[0],
                        shape=shape,
                    )

                samples_ddim, intermediates = sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=n_samples * len(prompt_list),
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    temperature=temperature,
                    x0=x0,
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim, )

                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0,
                    min=0.0,
                    max=1.0,
                )

                if save_result:
                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(
                            x_sample.cpu().numpy(),
                            'c h w -> h w c',
                        )

                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:04}.png"))

                    base_count += 1

                all_samples.append(x_samples_ddim, )

    grid = torch.stack(all_samples, 0)
    grid = rearrange(
        grid,
        'n b c h w -> (n b) c h w',
    )

    if num_rows is None:
        num_rows = 2

    grid = torchvision.utils.make_grid(
        grid,
        nrow=num_rows,
    )

    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    pil_img = Image.fromarray(grid.astype(np.uint8))

    if save_result:
        pil_img.save(
            os.path.join(
                outpath,
                f'{base_count}-{prompt_list[0].replace(" ", "-")}.png'))

        print(
            f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy."
        )

    return pil_img


if __name__ == "__main__":
    prompt_list = [
        "artstation artwork, psychedelic painting of a cat",
        "artstation artwork, psychedelic painting of a gorilla",
        "artstation artwork, psychedelic painting of a elephant",
        "artstation artwork, psychedelic painting of a lion",
    ]
    n_samples = 1
    #generate_from_prompt(prompt, plms=True, ddim_steps=10)
    #generate_from_prompt(prompt, plms=True, ddim_steps=50)
    #generate_from_prompt(prompt, plms=False, ddim_steps=10)
    img = generate_from_prompt(
        prompt_list,
        plms=False,
        ddim_steps=50,
        n_samples=n_samples,
        n_iter=1,
        num_rows=2,
    )
    img.save("output/ddim.png")

    img = generate_from_prompt(
        prompt_list,
        plms=True,
        ddim_steps=50,
        n_samples=n_samples,
        n_iter=1,
        num_rows=2,
    )
    img.save("output/plms.png")
