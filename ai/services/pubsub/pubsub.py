import abc
from typing import *


class PubSub(
        metaclass=abc.ABCMeta, ):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def subscribe(
        self,
        channel_name: str,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def publish(
        self,
        channel_name: str,
        data: Dict[str, Any],
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def listen(
        self,
        listen_cb: Callable,
    ):
        pass
