import os

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

    model.eval()

    # TODO: check if this is necessary
    model.to(device, )

    # HACK: can make stuff more efficient
    if mixed_precision:
        model.half()

    return model



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



def generate_from_prompt(
    prompt: str,
    ddim_steps: int = 200,
    ddim_eta: float = 0.0,
    plms: bool = False,
    n_iter: int = 1,
    H: int = 256,
    W: int = 256,
    n_samples: int = 4,
    temperature: float = 1.0,
    scale: float = 5.0,
):
    torch.manual_seed(0)
    base_count = len(os.listdir(sample_path))

    if plms:
        print("Using plms")
        if ddim_eta != 0:
            print("Warning! Using plms samples with a ddim_eta != 0")

        sampler = PLMSSampler(model, )

    else:
        sampler = DDIMSampler(model, )

    all_samples = list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None

            if scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * [""], )

            for _n in trange(n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(n_samples * [prompt], )
                shape = [4, H // 8, W // 8]
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    temperature=temperature,
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim, )
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0,
                    min=0.0,
                    max=1.0,
                )

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
    grid = torchvision.utils.make_grid(
        grid,
        nrow=n_samples,
    )

    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(
        os.path.join(outpath, f'{base_count}-{prompt.replace(" ", "-")}.png'))

    print(
        f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy."
    )


if __name__ == "__main__":
    prompt = "artstation artwork, psychedelic painting of a monkey"
    #generate_from_prompt(prompt, plms=True, ddim_steps=10)
    #generate_from_prompt(prompt, plms=True, ddim_steps=50)
    #generate_from_prompt(prompt, plms=False, ddim_steps=10)
    generate_from_prompt(prompt, plms=False, ddim_steps=100)

