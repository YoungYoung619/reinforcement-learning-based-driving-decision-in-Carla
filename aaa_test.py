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

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    world = client.load_world('Town04')
    pass