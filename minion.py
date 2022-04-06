import os
import logging
from threading import Thread
from typing import *

from ai.services.pubsub import PubSub, RedisPubSub, REDIS_HOST, REDIS_PORT
from ai.services.queue import Queue, RedisQueue
from ai.job_manager import JobManager
from ai.services.storage.s3 import S3, AWS_BUCKET_NAME, AWS_REGION_NAME

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
        self.queue.listen(
            channel_name=self.channel_name,
            cb=self.job_cb,
        )
        return


logger = logging.getLogger("minion")

channel_name = "ldm"
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

minion = Minion(
    channel_name=channel_name,
    pubsub=pubsub,
    queue=queue,
)

job_manager = JobManager(
    pubsub=pubsub,
    queue=queue,
    storage=storage,
    logger=logger,
    channel_name=channel_name,
)

minion.register_job(job_manager.process_message, )
minion.run()
