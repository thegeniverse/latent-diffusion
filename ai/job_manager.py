import json
import logging
from threading import Thread
from typing import *

from ai.image_utils import pil_to_base64
from ai.services.pubsub import PubSub
from ai.services.queue import Queue


class JobManager:
    def __init__(
        self,
        pubsub: PubSub,
        queue: Queue,
        storage,
        db=None,
        channel_name: str = "jobs",
        logger=logging,
    ):
        self.pubsub = pubsub
        self.queue = queue
        self.logger = logger
        self.storage = storage
        self.db = db
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

    def store_generations(
        self,
        data_dict,
    ):
        def storage_publisher(data_dict, ):
            print("Publishing!")
            print(data_dict)

            img = data_dict.get("img")
            user_id = data_dict.get("userId")
            generation_id = data_dict.get("generationId")
            prompt = data_dict.get("prompt")

            img_id = f"img-generations/{generation_id}.jpg"

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
                user_id,
                json.dumps(pub_data_dict),
            )
            print(f"Published in channel {user_id}")

            if self.db is not None:
                s3_img_url = f"https://{self.storage.bucket_name}.s3.{self.storage.region_name}.amazonaws.com/{img_id}"
                db_data_dict = dict(
                    imgURL=s3_img_url,
                    prompt=prompt,
                )
                self.db.store_generation(
                    user_id=user_id,
                    generation_id=generation_id,
                    generation_type="image",
                    data_dict=db_data_dict,
                )

        Thread(
            target=storage_publisher,
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

    def set_message_cb(
        self,
        cb: Callable,
    ):
        self.message_cb = cb

        return

    def process_message(
        self,
        message,
    ):
        print("processing message")
        print(f"data received: {message}")

        if message is None:
            return

        if isinstance(message, dict):
            job_data = json.loads(message, )

        elif isinstance(message, list):
            job_data = []
            for data in message:
                data_dict = json.loads(data, )
                job_data.append(data_dict, )

        else:
            print(f"ERROR! Unknown message data {type(message)}")

        result_list = self.message_cb(job_data, )

        for result_dict in result_list:
            self.store_generations(result_dict)

        return
