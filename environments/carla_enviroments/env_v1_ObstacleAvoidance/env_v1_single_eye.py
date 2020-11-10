"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

import sys
from environments.carla_enviroments.carla_config import base_config
try:
    sys.path.append(base_config.egg_file)
except IndexError:
    pass
import carla

from environments.carla_enviroments.carla_base import carla_base
from environments.carla_enviroments.utils import world_ops
from environments.carla_enviroments.utils import sensor_ops
from environments.carla_enviroments.env_v1_ObstacleAvoidance import env_v1_config
from environments.carla_enviroments.env_v1_ObstacleAvoidance import env_v1_two_eyes_config
from environments.carla_enviroments.env_v1_ObstacleAvoidance.test_scripts.generate_vehicles_pos import generate_vehicles_pos
from utilities.logging import logger
import os, random
import numpy as np
import math
from gym import spaces
import time as sys_time

from environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1 import ObstacleAvoidanceScenario

class ObstacleAvoidanceScenarioSingleEye(ObstacleAvoidanceScenario):
    def __init__(self):
        ObstacleAvoidanceScenario.__init__(self)

    def reattach_sensors(self):
        env_v1_config.collision_sensor_config['attach_to'] = self.ego
        self.collision_sensor = sensor_ops.collision_query(self.world, env_v1_config.collision_sensor_config)

        camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': env_v1_two_eyes_config.img_size[0],
                         'image_size_y': env_v1_two_eyes_config.img_size[1], 'fov': 110, 'sensor_tick': 0.02,
                         'transform': carla.Transform(carla.Location(x=1., y=0., z=2)),
                         'attach_to': self.ego}

        self.camera = sensor_ops.bgr_camera(self.world, camera_config)

    def get_imgs_from_camera(self):
        self.img = self.camera.get()
        return self.img

    def get_env_state(self):
        state = self.get_imgs_from_camera()
        return state

if __name__ == '__main__':
    import cv2

    scenario = ObstacleAvoidanceScenarioSingleEye()
    scenario.reset()
    i = 0
    while True:
        i += 1
        state, done = scenario.random_action_test_v2()

        cv2.imshow('img', state)
        cv2.waitKey(1)

        if done:
            scenario.reset()
            continue
    pass