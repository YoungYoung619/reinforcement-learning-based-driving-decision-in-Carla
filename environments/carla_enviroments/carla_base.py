import sys
import gym
from environments.carla_enviroments.carla_config import base_config
try:
    sys.path.append(base_config.egg_file)
except IndexError:
    pass
import carla
from utilities.logging import logger

class carla_base(gym.Env):
    """connect the carla server"""
    world = None ## carla world obj

    """connect the carla server"""
    def __init__(self):
        try:
            client = carla.Client('127.0.0.1', 2000)
            logger.info('carla connecting...')
            client.set_timeout(2.0)
            self.world = client.get_world()
        except:
            logger.error('carla connection fail...')
        else:
            logger.info('carla connection success...')

if __name__ == '__main__':
    a = carla_base()
    pass