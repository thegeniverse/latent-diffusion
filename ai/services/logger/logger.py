import logging

logging.basicConfig(
    # filename='logs.txt',
    format="%(levelname)s:%(message)s",
    level=logging.DEBUG,
)


def get_error_type_from_str(error_str: str, ):
    """
    Compute the type of error represented in `error_str`. Error codes:
    - cuda-oom --> cuda out of memory
    
    Args:
        error_str (str): error string

    Returns:
        str: resulting error code
    """
    error_type = "unknown"

    if "CUDA out of memory" in error_str:
        error_type = "cuda-oom"


    if "The size of tensor a" in error_str and \
        "must match the size of tensor b" in error_str:
        error_type = "tensor-size-mismatch"

    return error_type


class Logger(
        logging.RootLogger, ):
    def __init__(self, ):
        super(Logger, self).__init__(logging.WARNING)
        self.setConsoleLevel(logging.WARNING, )

    def exception(
        self,
        exception: Exception,
    ):
        error_str = str(exception.args[0])

        error_type = get_error_type_from_str(error_str, )

        self.error(error_type)

        return error_type
