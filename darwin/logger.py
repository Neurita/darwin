
import os
import yaml
import logging.config

from .config import LOG_LEVEL


def setup_logging(default_path=os.path.join(os.path.dirname(__file__),
                                            'logger.yaml'),
                  default_level=LOG_LEVEL,
                  env_key='DARWIN_LOG_CFG'):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f)
        logging.config.dictConfig(config)
        #print('Started logging using config file {0}.'.format(path))
    else:
        logging.basicConfig(level=default_level)
        #print('Started default logging. Could not find config file '
        #      'in {0}.'.format(path))
    log = logging.getLogger(__name__)
    log.info('Start logging.')
