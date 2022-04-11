import os
import logging
from threading import Thread
from typing import *

import torchvision

from ai.services.pubsub import PubSub, RedisPubSub, REDIS_HOST, REDIS_PORT
from ai.services.queue import Queue, RedisQueue
from ai.job_manager import JobManager
from ai.services.storage.s3 import S3, AWS_BUCKET_NAME, AWS_REGION_NAME
from ai.services.db.firestore import Firestore, FIREBASE_CREDENTIALS, PROJECT_ID
from generation_utils import LatentDiffusionModel

PUBSUB_HOST = os.getenv("PUBSUB_HOST", REDIS_HOST)
PUBSUB_PORT = os.getenv("PUBSUB_PORT", REDIS_PORT)

QUEUE_HOST = os.getenv("QUEUE_HOST", REDIS_HOST)
QUEUE_PORT = os.getenv("PUBSUB_PORT", REDIS_PORT)

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", AWS_BUCKET_NAME)
REGION_NAME = os.getenv("AWS_REGION_NAME", AWS_REGION_NAME)


class Minion:
    def __init__(
        self,
        channel_name: str,
        pubsub: PubSub,
        queue: Queue,
    ):
        self.channel_name = channel_name
        self.pubsub = pubsub
        self.queue = queue

        self.job_cb = None

    def register_job(
        self,
        cb: Callable,
    ):
        self.job_cb = cb

    def start(self, ):
        thread = Thread(
            target=self.pubsub.subscribe,
            kwargs=dict(
                channel_name=self.channel_name,
                cb=self.job_cb,
            ),
        )

        thread.start()

        return thread

    def run(self, ):
        # self.pubsub.subscribe(
        #     channel_name=self.channel_name,
        #     cb=self.job_cb,
        # )
        print("minion running")
        print(self.channel_name)
        self.queue.listen(
            channel_name=self.channel_name,
            cb=self.job_cb,
        )
        return


if __name__ == "__main__":

    latent_diffusion_model = LatentDiffusionModel()

    def message_cb(job_data_list, ):
        prompt_list = []
        generation_id_list = []
        user_id_list = []

        result_list = []

        for job_data_dict in job_data_list:
            num_generations = job_data_dict.get("numGenerations")
            if num_generations is not None:
                prompt_list += job_data_dict.get("prompt")
                generation_id_list += job_data_dict.get("generationId")
                user_id_list += job_data_dict.get("userId")

            else:
                prompt_list.append(job_data_dict.get("prompt"))
                generation_id_list.append(job_data_dict.get("generationId"))
                user_id_list.append(job_data_dict.get("userId"))

        result_tensor_imgs = latent_diffusion_model.generate_from_prompt(
            prompt_list, )[0]

        for result_idx, result_tensor in enumerate(result_tensor_imgs):
            img_pil = torchvision.transforms.ToPILImage()(result_tensor, )

            result_list.append(
                {
                    "prompt": prompt_list[result_idx],
                    "generationId": generation_id_list[result_idx],
                    "userId": user_id_list[result_idx],
                    "img": img_pil,
                }, )

        return result_list

    logger = logging.getLogger("minion")

    channel_name = "generations"
    pubsub = RedisPubSub(
        host=PUBSUB_HOST,
        port=REDIS_PORT,
        logger=logger,
    )
    queue = RedisQueue(
        host=QUEUE_HOST,
        port=QUEUE_PORT,
        logger=logger,
    )
    storage = S3(
        bucket_name=BUCKET_NAME,
        region_name=REGION_NAME,
    )

    db = Firestore(
        firebase_credentials_path=FIREBASE_CREDENTIALS,
        project_id=PROJECT_ID,
    )

    minion = Minion(
        channel_name=channel_name,
        pubsub=pubsub,
        queue=queue,
    )

    job_manager = JobManager(
        pubsub=pubsub,
        queue=queue,
        storage=storage,
        db=db,
        logger=logger,
        channel_name=channel_name,
    )
    job_manager.set_message_cb(message_cb, )

    minion.register_job(job_manager.process_message, )
    print("Minion running!")
    minion.run()
