import base64
import os
import subprocess
import uuid
from io import BytesIO
from typing import *

from PIL import Image

from ai.config import HTTP_URL


def base64_to_PIL(base64_encoding: str):
    return Image.open(BytesIO(base64.b64decode(base64_encoding)))


def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode('ascii')

    return data_uri


def generate_video_from_img_list(
    user_id: str,
    img_list: List[Image.Image],
    prefix: str = "",
    fps=16,
):
    results_dir = os.path.join(
        "results",
        user_id,
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    video_id = str(uuid.uuid4())
    filename = f"{prefix}{video_id}"
    thumbnail_filename = f"{filename}.jpg"
    video_filename = f"{filename}.mp4"

    out_thumbnail_path = os.path.join(
        results_dir,
        f"{thumbnail_filename}",
    )

    out_video_path = os.path.join(
        results_dir,
        f"{video_filename}",
    )

    img_list[0].save(out_thumbnail_path)

    for idx, img in enumerate(img_list):
        out_img_path = os.path.join(
            results_dir,
            f"{str(idx).zfill(6)}.jpg",
        )
        img.save(out_img_path)

    cmd = ("ffmpeg -y "
           "-r 16 "
           f"-pattern_type glob -i '{results_dir}/0*.jpg' "
           "-vcodec libx264 "
           f"-crf {fps} "
           "-pix_fmt yuv420p "
           f"{out_video_path};"
           f"rm -r {results_dir}/0*.jpg;")

    subprocess.check_call(cmd, shell=True)

    # video_url = os.path.join(
    #     HTTP_URL,
    #     out_video_path,
    # )
    # thumbnail_url = os.path.join(
    #     HTTP_URL,
    #     out_thumbnail_path,
    # )

    return out_video_path, out_thumbnail_path
