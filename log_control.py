"""
log_control.py

This file provides loggers

- Rakshit Agrawal, 2017
"""

LOGGER_TYPE = 'tensorflow'

if LOGGER_TYPE == 'tensorflow':
    from tensorflow import logging

    logi = logging.info
    logd = logging.debug
    logw = logging.warning
    loge = logging.error
    logging.set_verbosity(logging.INFO)

    log_levels = {
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'WARN': logging.WARN,
        'WARNING': logging.WARN,
        'ERROR': logging.ERROR
    }


    def set_log_level(level):
        logging.set_verbosity(log_levels.get(level.upper(), "INFO"))

elif LOGGER_TYPE == 'python':
    import logging
    import sys

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger = logging.getLogger()
    logi = logger.info
    logd = logger.debug
    logw = logger.warning
    loge = logger.error

    log_levels = {
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'WARN': logging.WARN,
        'WARNING': logging.WARN,
        'ERROR': logging.ERROR
    }


    def set_log_level(level):
        logger.setLevel(log_levels.get(level.upper(), "INFO"))
