import sys
import random
import numpy as np

from environments.carla_enviroments.carla_config import base_config
try:
    sys.path.append(base_config.egg_file)
except IndexError:
    pass
import carla

from environments.carla_enviroments.env_v1_ObstacleAvoidance.test_scripts.generate_vehicles_pos import generate_vehicles_pos
from environments.carla_enviroments.utils import world_ops
from environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1 import ObstacleAvoidanceScenario
from environments.carla_enviroments.env_v1_ObstacleAvoidance import env_v1_config

class ObstacleAvoidanceScenarioDynamic(ObstacleAvoidanceScenario):

    def __init__(self):
        ObstacleAvoidanceScenario.__init__(self)
        self.init_state = True

    def respawn_vehicles(self):
        only_one_vehicle = False

        if not only_one_vehicle:
            if not env_v1_config.fix_vehicle_pos:
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(12, 12))

            obstacles = []
            n_vehicle= len(self.vehicles_pos)
            ego_idx = random.randint(0, int(n_vehicle*0.4))
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
        for other in self.obstacles:
            other.set_autopilot(True)

    def step(self, action_idx):
        state, reward, done, _ = ObstacleAvoidanceScenario.step(self, action_idx)

        if self.init_state:
            self.make_others_autopilot()    ## 等确保自车动了之后，其他车才开始动，不然其他车会走很远一段距离，自车才开始动
            self.init_state = False
        return state, reward, done, _

    def get_obstacles_state(self, ego, obstacles):
        state = ObstacleAvoidanceScenario.get_obstacles_state(self, ego, obstacles) ## the position and exist_flag

        ego_velocity = ego.get_velocity()
        left_velocity = self.left_obstacle.get_velocity()
        right_velocity = self.right_obstacle.get_velocity()

        left_ego_v_x = (ego_velocity.x - left_velocity.x) / self.max_longitude_velocity
        left_ego_v_y = (ego_velocity.y - left_velocity.y) / self.max_lateral_velocity

        right_ego_v_x = (ego_velocity.x - right_velocity.x) / self.max_longitude_velocity
        right_ego_v_y = (ego_velocity.y - right_velocity.y) / self.max_lateral_velocity

        state = state + [left_ego_v_x, left_ego_v_y, right_ego_v_x, right_ego_v_y]
        return state


if __name__ == '__main__':

    scenario =ObstacleAvoidanceScenarioDynamic()
    scenario.reset()
    t = True
    while True:
        _, done = scenario.random_action_test_v2()

        if done:
            scenario.reset()
            continue