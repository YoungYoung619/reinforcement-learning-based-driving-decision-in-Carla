'''generate the vehicle position in carla world for this scenerio'''
import random
import numpy as np

import sys
from environments.carla_enviroments.carla_config import base_config
try:
    sys.path.append(base_config.egg_file)
except IndexError:
    pass
import carla

import time
from environments.carla_enviroments.carla_base import carla_base
from environments.carla_enviroments.utils import world_ops
from environments.carla_enviroments.utils import sensor_ops
from environments.carla_enviroments.env_v1_ObstacleAvoidance import env_v1_config
from utilities.logging import logger

road_range = {'x':(-380., 50.), 'y':(13., 16.5)}
# road_range = {'x':(-380., 50.), 'y':(13., 16.5)}

# road_range = {'x':(-380., 50.), 'y':(11.2, 11.2)}
# road_range = {'x':(-380., 50.), 'y':(18.1, 18.1)}

# left_lane_center_formula = lambda x: (x - -428.)/(100. - -428.) * (16.5 - 16.5) + 16.5
# right_lane_center_formula = lambda x: (x - -428.)/(100. - -428.) * (13 - 13) + 13

def generate_vehicles_pos(n_vehicles):
    """ random spawn an autopilot at a transform in carla world
    Args:
        world: carla world instance
        distance_freq: a distance control the vehicle number in this road area
    """
    x_min = road_range['x'][0]
    x_max = road_range['x'][1]
    x_dis = x_max - x_min
    distance_freq = x_dis // n_vehicles
    vehicle_pos = []
    for i in range(n_vehicles):
        x = random.uniform(x_min + 10. + i*distance_freq, x_min + 10. + (i+1)*distance_freq - 10.)
        y = random.sample(road_range['y'], 1)[0]
        if x < -250:
            z = 5.
        else:
            z = 15.
        vehicle_pos.append(np.array([x, y, z]))
        # point.rotation.yaw = -0.142975

    return vehicle_pos


if __name__ == '__main__':
    # for n_vehicle in range(4, 20):
    #     vehicle_pos = generate_vehicles_pos(n_vehicle)
    #     npz_path = '..\saves\positions\\%d_vehicle_positions.npz'%(n_vehicle)
    #     np.savez(npz_path, pos=vehicle_pos)
    #     pass
    pass
    # client = carla.Client('localhost', 2000)
    # world = client.load_world('Town04')
    #
    # vehicle_pos = generate_vehicles_pos(27)
    #
    # for idx, vehicle_pos in enumerate(vehicle_pos):
    #     transform = carla.Transform()
    #     transform.location.x = vehicle_pos[0]
    #     transform.location.y = vehicle_pos[1]
    #     transform.location.z = vehicle_pos[2]
    #     transform.rotation.yaw = -0.142975
    #     if idx == 0:  ## ego
    #         ego = world_ops.try_spawn_random_vehicle_at(world=world, transform=transform,
    #                                                          role_name='ego', autopilot=False,
    #                                                          vehicle_type='vehicle.tesla.model3')
    #         time.sleep(0.02)
    #     else:
    #         obstacle = world_ops.try_spawn_random_vehicle_at(world=world, transform=transform,
    #                                                          role_name='other', autopilot=False)
    #         time.sleep(0.02)

