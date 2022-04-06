import os
import logging

# NOTE: "prod" or "dev"
HTTP_PORT = os.getenv("HTTP_PORT", default=8001)
SERVER_IP = os.getenv("SERVER_IP", default="localhost")
HTTP_URL = f"https://{SERVER_IP}:{HTTP_PORT}"

if SERVER_IP == "localhost":
    logging.warning(f"`SERVER_IP` not defined. Using {SERVER_IP}")

# TODO: remove this shit
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
