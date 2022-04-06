import logging
from typing import *

from redis import Redis

from ai.services.pubsub import PubSub


class RedisPubSub(PubSub):
    def __init__(
        self,
        host: str,
        port: int,
        logger=logging,
    ) -> None:
        self.logger = logger

        self.redis_client = Redis(
            host=host,
            port=port,
        )

        self.pubsub = self.redis_client.pubsub()

        return

    def publish(
        self,
        channel_name: str,
        data: Dict[str, Any],
    ) -> None:
        # XXX data might need to be in bytes
        self.redis_client.publish(
            channel_name,
            data,
        )

        return

    def subscribe(
        self,
        channel_name: str,
        cb: Callable,
    ) -> None:
        # XXX check if subscribe returns something
        _ = self.pubsub.subscribe(channel_name, )

        self.listen(cb, )

        return

    def listen(
        self,
        listen_cb: Callable,
    ) -> None:
        self.logger.debug("Redis listening...")
        for message in self.pubsub.listen():
            self.logger.debug(f"New message: `{message}`")

            if "type" in message and message["type"] == 'subscribe':
                print("Handshake received.")
                continue

            print(f"New message: {message}")

            listen_cb(message, )

            self.logger.debug("Redis listening...")

        return
