import logging
import os

import yaml

logger = logging.getLogger(__name__)


def get_env():
    env = None

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_dir = os.path.join(base_dir, 'environments')

    logger.debug(f'Environment settings from: {env_dir}')

    for fn in os.listdir(env_dir):
        if fn.endswith('.yml'):
            with open(os.path.join(env_dir, fn), 'r') as f:
                envs = yaml.load(f, Loader=yaml.SafeLoader)

                for env_name, _env in envs.items():
                    if os.path.exists(_env['must_exists']):
                        logger.info(f'Environment detected: {env_name} (in {fn})')
                        env = _env
                        break
        if env is not None:
            break

    if env:
        return env
    else:
        raise ValueError('Could not determine env!')
