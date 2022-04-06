import json
import logging
import uuid
from threading import Thread
from typing import *

from ai.image_utils import pil_to_base64
from ai.services.pubsub import PubSub
from ai.services.queue import Queue
from generation_utils import generate_from_prompt


class JobManager:
    def __init__(
        self,
        pubsub: PubSub,
        queue: Queue,
        storage,
        channel_name: str = "jobs",
        logger=logging,
    ):
        self.pubsub = pubsub
        self.queue = queue
        self.logger = logger
        self.storage = storage
        self.channel_name = channel_name

        self.modeling = None

    def publisher(
        self,
        job_data_dict: Dict[str, Any],
    ):
        user_id = job_data_dict["userId"]

        img = job_data_dict.get('img')
        if img is not None:
            job_data_dict["img"] = pil_to_base64(img, )

        _ = self.pubsub.publish(
            user_id,
            json.dumps(job_data_dict),
        )

        job_data_dict = self.queue.rpop(f"{user_id}", )

        if job_data_dict is None:
            return

        return

    def s3_publisher_cb(
        self,
        data_dict,
    ):
        def s3_publisher(data_dict, ):
            print("Publishing to s3!")

            prompt = data_dict.get("prompt")
            step = data_dict.get("step")
            event = data_dict["event"]
            img = data_dict.get("img")
            thumbnail = data_dict.get("thumbnail")
            video = data_dict.get("video")
            user_id = data_dict["userId"]

            if step is None:
                step = 0

            job_id = str(uuid.uuid4())

            if prompt is not None:
                element_id = f"{job_id}_{prompt}_{step}"

            else:
                element_id = f"{job_id}_{step}"

            thumbnail_id = f"{event}/{element_id}_thumbnail.jpg"
            img_id = f"{event}/{element_id}_img.jpg"
            video_id = f"{event}/{element_id}_video.mp4"

            if img is not None:
                self.storage.upload_pil_img(
                    pil_img=img,
                    img_id=img_id,
                    img_format="JPEG",
                )

            pub_data_dict = dict(
                event=data_dict.get("event"),
                imgId=img_id,
                step=data_dict.get("step"),
                bucketName=self.storage.bucket_name,
                regionName=self.storage.region_name,
            )

            _ = self.pubsub.publish(
                user_id,
                json.dumps(pub_data_dict),
            )

            if thumbnail is not None:
                self.storage.upload_pil_img(
                    pil_img=thumbnail,
                    img_id=thumbnail_id,
                    img_format="JPEG",
                )

            if video is not None:
                self.storage.upload_video(
                    video=video,
                    video_id=video_id,
                )

            print(f"Data from step {data_dict['step']} in aws ({user_id})")

        Thread(
            target=s3_publisher,
            args=(data_dict, ),
        ).start()

    def error_cb(
        self,
        data_dict: Dict[str, Any],
    ):
        return

    def step_cb(
        self,
        job_data_dict: Dict[str, Any],
    ):
        return

    def process_message(
        self,
        *args,
        **kwargs,
    ):
        print("processing message")
        job_data_dict = self.queue.rpop(self.channel_name)

        print(f"data received: {job_data_dict}")

        if job_data_dict is None:
            return

        job_data_dict = json.loads(job_data_dict, )

        prompt = job_data_dict.get("prompt")
        job_id = job_data_dict.get("jobId")
        img_pil = generate_from_prompt(prompt, )
        self.s3_publisher_cb({
            "userId": job_id,
            "img": img_pil,
        })

        return
