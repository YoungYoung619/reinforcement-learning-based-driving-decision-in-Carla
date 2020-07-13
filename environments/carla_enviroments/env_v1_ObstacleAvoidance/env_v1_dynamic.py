import sys
import random
import numpy as np

from environments.carla_enviroments.carla_config import base_config
try:
    sys.path.append(base_config.egg_file)
except IndexError:
    pass
import carla
from environments.carla_enviroments.utils import sensor_ops
from environments.carla_enviroments.env_v1_ObstacleAvoidance.test_scripts.generate_vehicles_pos import generate_vehicles_pos
from environments.carla_enviroments.utils import world_ops
from environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1 import ObstacleAvoidanceScenario
from environments.carla_enviroments.env_v1_ObstacleAvoidance import env_v1_config

class ObstacleAvoidanceScenarioDynamic(ObstacleAvoidanceScenario):

    def __init__(self):
        ObstacleAvoidanceScenario.__init__(self)
        self.init_state = True

        env_v1_config.actions = {0:[0.6, 0.5, 0.],
                                 1:[0.6, -0.5, 0.],
                                 2:[0.6, 0.1, 0.],
                                 3:[0.6, -0.1, 0.],
                                 4:[0.6, 0., 0.]}    # small acc

    def respawn_vehicles(self):
        only_one_vehicle = False

        if not only_one_vehicle:
            if not env_v1_config.fix_vehicle_pos:
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(23, 23))

            obstacles = []
            n_vehicle= len(self.vehicles_pos)
            ego_idx = 0
            for idx, vehicle_pos in enumerate(self.vehicles_pos):
                transform = carla.Transform()
                transform.location.x = vehicle_pos[0]
                transform.location.y = vehicle_pos[1]
                transform.location.z = vehicle_pos[2]
                transform.rotation.yaw = -0.142975
                if idx == ego_idx:  ## ego
                    self.ego = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                                     role_name='ego', autopilot=False,
                                                                     vehicle_type='vehicle.tesla.model3')
                    self.wait_for_response()
                    pass
                else:
                    obstacle = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                                     role_name='other', autopilot=False)
                    self.wait_for_response()
                    obstacles.append(obstacle)
            self.obstacles = obstacles
        else:
            vehicle_pos = [-30.,
                           (env_v1_config.lateral_pos_limitation[0] + env_v1_config.lateral_pos_limitation[1]) / 2.,
                           1.81]
            transform = carla.Transform()
            transform.location.x = vehicle_pos[0]
            transform.location.y = vehicle_pos[1]
            transform.location.z = vehicle_pos[2]
            transform.rotation.yaw = -0.142975

            self.ego = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                             role_name='ego', autopilot=False,
                                                             vehicle_type='vehicle.tesla.model3')
            self.wait_for_response()
            self.obstacles = []

    def reset(self):
        ObstacleAvoidanceScenario.reset(self)
        self.init_state = True
        return self.get_env_state()

    def make_others_autopilot(self):
        throttle = random.uniform(0.2, 0.4)
        for other in self.obstacles:
            # other.set_autopilot(True)
            other.apply_control(carla.VehicleControl(throttle=throttle,
                                                    steer=0., brake=0.))

    def step(self, action_idx):
        """conduct action in env
                Args:
                    action: int, an idx
                Return: env state, reward, done, info
                """
        # --- conduct action and holding a while--- #
        if self.test_mode:
            action = np.array(env_v1_config.actions[action_idx])
            action = 0.5 * action + 0.5 * self.last_action
            self.last_action = action
        else:
            action = env_v1_config.actions[action_idx]

        self.resume()

        self.ego.apply_control(carla.VehicleControl(throttle=action[0],
                                                    steer=action[1], brake=action[2]))
        self.wait_env_running(time=env_v1_config.action_holding_time)

        # -- next state -- #
        state = self.get_env_state()
        self.pause()

        # --- reward --- # forward distance, velocity and center pos
        # forward_distance = state[0]
        # velocity = math.sqrt(state[3]**2 + state[4]**2 + 1e-8)
        # lateral_pos = state[1]
        # reward = self.__get_reward_v1(forward_distance=forward_distance, velocity=velocity,
        #                               lateral_pos=lateral_pos)
        # reward = self.__get_reward_v1()
        reward = self.get_reward_v4(state=state)
        # --reset some var -- #
        # self.last_forward_distance = forward_distance

        self.step_counter += 1

        done = self.is_done()

        if self.init_state:
            self.make_others_autopilot()    ## 等确保自车动了之后，其他车才开始动，不然其他车会走很远一段距离，自车才开始动
            self.init_state = False
        return state, reward, done, {}

    def attach_camera_to_ego(self):
        camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': 640,
                              'image_size_y': 360, 'fov': 110, 'sensor_tick': 0.02,
                              'transform': carla.Transform(carla.Location(x=-0., y=-0.4, z=1.25)),
                              'attach_to': self.ego}

        self.camera = sensor_ops.bgr_camera(self.world, camera_config)



if __name__ == '__main__':

    scenario = ObstacleAvoidanceScenarioDynamic()
    scenario.reset()
    t = True
    while True:
        _, done = scenario.random_action_test_v2()

        if done:
            scenario.reset()
            continue