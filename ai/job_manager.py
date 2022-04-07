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
            print(data_dict)

            img = data_dict.get("img")
            job_id = data_dict.get("jobId")

            element_id = f"{job_id}"

            img_id = f"{element_id}.jpg"

            self.storage.upload_pil_img(
                pil_img=img,
                img_id=img_id,
                img_format="JPEG",
            )
            
            print(f"Data in aws!")

            pub_data_dict = dict(
                imgId=img_id,
                bucketName=self.storage.bucket_name,
                regionName=self.storage.region_name,
            )

            _ = self.pubsub.publish(
                job_id,
                json.dumps(pub_data_dict),
            )
            print(f"Published in channel {job_id}")


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

        prompt = job_data_dict.get("text")
        job_id = job_data_dict.get("id")
        img_pil = generate_from_prompt(prompt, )
        self.s3_publisher_cb({
            "jobId": job_id,
            "img": img_pil,
        })

        return
