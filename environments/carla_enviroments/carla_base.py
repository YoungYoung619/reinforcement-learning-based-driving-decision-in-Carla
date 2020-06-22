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
            self.client = carla.Client('127.0.0.1', 2000)
            logger.info('carla connecting...')
            self.client.set_timeout(2.0)
        except:
            raise RuntimeError('carla connection fail...')
        else:
            logger.info('carla connection success...')

    def start_synchronous_mode(self):
        """carla synchoronous mode"""
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=base_config.no_render_mode,
                                                      synchronous_mode=True))

    def close_synchronous_mode(self):
        """close synchoronous mode"""
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=base_config.no_render_mode,
                                                      synchronous_mode=False))

    def wait_carla_runing(self, time):
        """Wait for Carla to run for a specific time"""
        time_elapse = 0.
        while True:
            time_elapse += self.wait_for_response()
            if time_elapse > time:
                break

    def wait_for_response(self):
        """wait for carla response
        Return:
            response time consumption
        """
        self.world.tick()
        elapse_time = self.world.wait_for_tick()
        # logger.info('respond time consumption %f'%(round(elapse_time.delta_seconds, 6)))
        return elapse_time.delta_seconds

    def pause(self):
        """pause the simulator"""
        self.start_synchronous_mode()

    def resume(self):
        """resume the simulator from pause"""
        self.close_synchronous_mode()

if __name__ == '__main__':
    a = carla_base()
    pass