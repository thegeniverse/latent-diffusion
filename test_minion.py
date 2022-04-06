import json
from typing import *

from redis import Redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379

BUCKET_NAME = "genibot"
REGION_NAME = "us-west-2"

channel_name = "ldm"

redis_client = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
)
redis_client.flushdb()

data_dict = dict(prompt="kawaii pixel art corgi", )

job_id = "666"
data = json.dumps(data_dict, )

redis_client.lpush(
    channel_name,
    data,
)
print('data pushed')
print(channel_name)
