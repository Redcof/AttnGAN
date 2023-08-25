import logging

# create logger object
logger = None
# create a formatter object
Log_Format = "%(asctime)s %(name)s [%(levelname)s]: %(message)s"
formatter = logging.Formatter(fmt=Log_Format)


def init_logger(name="attgan", log_file=None):
    global logger
    logger = logging.getLogger(name)
    
    # Add custom handler with format to this logger
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    if log_file:
        attach_file_to_logger(log_file)


def attach_file_to_logger(log_file):
    """This function attaches file stream to the logger to consume all logged information and saved it"""
    global logger
    # Creates a new logs every minute if restarted.
    # Otherwise, it will use the same file
    handler = logging.FileHandler(log_file, "a")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return handler
