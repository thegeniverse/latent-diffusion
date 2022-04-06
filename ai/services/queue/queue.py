import abc
from typing import *


class Queue(
        metaclass=abc.ABCMeta, ):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def lpush(
        self,
        *args,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def rpop(
        self,
        *args,
        **kwargs,
    ):
        pass
