import logging
import json
import time
from typing import *

from redis import Redis


class RedisQueue:
    def __init__(
        self,
        host: str,
        port: int,
        logger=logging,
    ):
        self.logger = logger
        self.redis_client = Redis(
            host=host,
            port=port,
        )

    def lpush(
        self,
        channel_name: str,
        data: Dict[str, Any],
    ):
        self.redis_client.lpush(
            channel_name,
            data,
        )

    def rpop(
        self,
        channel_name: str,
    ):
        return self.redis_client.rpop(channel_name, )

    def lpop(
        self,
        channel_name: str,
    ):
        return self.redis_client.lpop(channel_name, )

    def listen(
        self,
        channel_name: str,
        cb: Callable,
    ) -> None:
        self.logger.debug("Redis queue listening...")

        while True:
            # message = self.rpop(channel_name, )
            try:
                cb(None, )

            except Exception as e:
                self.logger.info("Error in queue listen.", )
                self.logger.info(e, )

            time.sleep(0.5)

            self.logger.debug("Redis listening...")
