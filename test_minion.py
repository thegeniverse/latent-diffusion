import time
import json
from typing import *

from redis import Redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379

BUCKET_NAME = "genibot"
REGION_NAME = "us-west-2"

queue_name = "generations"

redis_client = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
)
redis_client.flushdb()

data_dict = dict(
    prompt="kawaii pixel art corgi",
    userId="696969",
    generationId="1",
)
data = json.dumps(data_dict, )

redis_client.lpush(
    queue_name,
    data,
)
print('data pushed')
print(queue_name)

data_dict = dict(
    prompt="kawaii pixel art corgi",
    userId="420420",
    generationId="2",
)
data = json.dumps(data_dict, )

redis_client.lpush(
    queue_name,
    data,
)
print('data pushed')
print(queue_name)
