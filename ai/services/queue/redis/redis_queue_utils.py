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

    def rpop_buffered(
        self,
        channel_name: str,
        buffer_size: int,
    ):

        while True:
            message = self.rpop(channel_name)
            print("listening...")

            if message is not None:
                print("message received!")
                print(message)

                message_list = [message, ]

                for _ in range(buffer_size):
                    message= self.rpop(channel_name)
                    if message is not None:
                        message_list.append(message)
                    
                        time.sleep(0.05)

                return message_list

            else:
                time.sleep(0.1)

        return 


    def listen(
        self,
        channel_name: str,
        cb: Callable,
        buffer_size: int = 16,
    ) -> None:
        print("listening...")

        while True:
            job_data_list = self.rpop_buffered(
                channel_name,
                buffer_size,
            )
            
            print("job data")
            print(job_data_list)

            if len(job_data_list):
                try:
                    print("processing message")
                    cb(job_data_list, )

                except Exception as e:
                    print("Error in queue listen.", )
                    print(e, )

            time.sleep(0.1)

            print("Redis listening...")
