import io
import logging

import torch
import numpy as np
from PIL import Image

import boto3


class S3:
    def __init__(
        self,
        bucket_name: str,
        region_name: str,
        logger=logging,
    ):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.logger = logger

        self.client_s3 = boto3.client(
            's3',
            region_name=self.region_name,
        )

    def upload_pil_img(
        self,
        pil_img: Image.Image,
        img_id: str,
        img_format: str = "JPEG",
        watermark: bool = True,
    ):
        if watermark:
            watermark_downscale_factor = 8

            pil_watermark = Image.open("assets/watermark.png", ).resize((
                int(pil_img.size[0] / watermark_downscale_factor),
                int(pil_img.size[1] / watermark_downscale_factor),
            ))

            pil_img.paste(
                pil_watermark,
                (pil_img.size[0] - pil_watermark.size[0],
                 pil_img.size[1] - pil_watermark.size[1]),
            )

            # watermark_transparency = 60
            # watermark_downscale_factor = 8

            # pil_watermark = Image.open("assets/watermark.png", ).resize((
            #     int(pil_img.size[0] / watermark_downscale_factor),
            #     int(pil_img.size[1] / watermark_downscale_factor),
            # ))

            # alpha = Image.new("L", pil_watermark.size, 255)
            # pil_watermark.putalpha(alpha, )

            # paste_mask = pil_watermark.split()[3].point(
            #     lambda i: i * watermark_transparency / 100.)

            # pil_img.paste(
            #     pil_watermark,
            #     (pil_img.size[0] - pil_watermark.size[0],
            #      pil_img.size[1] - pil_watermark.size[1]),
            #     mask=paste_mask,
            # )

        buffer = io.BytesIO()

        pil_img.save(
            buffer,
            format=img_format,
        )
        buffer.seek(0)

        self.client_s3.upload_fileobj(
            buffer,
            self.bucket_name,
            img_id,
        )

        return

    def upload_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: str,
    ):
        tensor = tensor.detach().clone()
        buffer = io.BytesIO()
        torch.save(
            tensor,
            buffer,
        )
        buffer.seek(0)

        self.client_s3.upload_fileobj(
            buffer,
            self.bucket_name,
            tensor_id,
        )

        return

    def upload_video(
        self,
        video: str,
        video_id: str,
    ):
        self.client_s3.upload_file(
            video,
            self.bucket_name,
            video_id,
        )

        return

    def get_object_from_s3(
        self,
        object_id,
    ):
        s3_response_object = self.client_s3.get_object(
            Bucket=self.bucket_name,
            Key=object_id,
        )
        object_content = s3_response_object['Body'].read()

        return object_content

    def download_pil(
        self,
        img_id: str,
    ):
        img_object = self.get_object_from_s3(object_id=img_id, )

        buffer = io.BytesIO(img_object, )
        img = Image.open(buffer, )

        return img

    def download_tensor(
        self,
        tensor_id: str,
    ):
        tensor_object = self.get_object_from_s3(object_id=tensor_id, )

        buffer = io.BytesIO(tensor_object, )
        latents = torch.tensor(torch.load(buffer, ))

        return latents


if __name__ == "__main__":
    storage = S3(
        bucket_name="geniverse",
        region_name="us-west-2",
    )

    storage.upload_video(
        "test_results/696969/interpolation-8ec3a58a-9b85-464d-bc41-caa633d4141b.mp4",
        "vid.mp4",
    )
