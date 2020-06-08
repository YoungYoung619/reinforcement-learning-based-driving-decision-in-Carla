'''generate the vehicle position in carla world for this scenerio'''
import random
import numpy as np

road_range = {'x':(-40., 154.), 'y':(204., 207.5)}

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
        x = random.randint(x_min + 10. + i*distance_freq, x_min + 10. + (i+1)*distance_freq - 10.)
        y = random.sample(road_range['y'], 1)[0]
        if x > 120:
            z = 3.
        else:
            z = 1.81
        vehicle_pos.append(np.array([x, y, z]))
        # point.rotation.yaw = -0.142975

    return vehicle_pos


if __name__ == '__main__':
    for n_vehicle in range(4, 20):
        vehicle_pos = generate_vehicles_pos(n_vehicle)
        npz_path = '..\saves\positions\\%d_vehicle_positions.npz'%(n_vehicle)
        np.savez(npz_path, pos=vehicle_pos)
        pass
