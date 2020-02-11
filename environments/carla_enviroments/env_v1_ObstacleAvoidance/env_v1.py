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
from environments.carla_enviroments.env_v1_ObstacleAvoidance.test_scripts.generate_vehicles_pos import generate_vehicles_pos
from utilities.logging import logger
import time, os, random
import numpy as np
import math

class ObstacleAvoidanceScenario(carla_base):
    def __init__(self):
        carla_base.__init__(self)

        if env_v1_config.fix_vehicle_pos:
            ## using predefine vehicles position
            self.vehicles_pos = np.load(env_v1_config.vehicles_pos_file)['pos']
        else:
            ## random generate vehicles position
            self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(5, 10))

        self.start_synchronous_mode()

    def step(self, action):
        pass

    def reset(self):
        """reset the world"""
        world_ops.destroy_all_actors(self.world)
        self.__wait_for_response()
        self.__respawn_vehicles()
        self.__reattach_sensors()

        ## Waiting for the vehicles to land on the ground
        time_elapse = 0.
        while True:
            time_elapse += self.__wait_for_response()
            if time_elapse > 0.8:
                break

    def start_synchronous_mode(self):
        """carla synchoronous mode"""
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False,
                                                      synchronous_mode=True))

    def close_synchronous_mode(self):
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False,
                                                      synchronous_mode=False))

    def __get_env_state(self):
        def get_ego_state(ego):
            ego_transform = ego.get_transform()
            ego_velocity = ego.get_velocity()
            ego_angular = ego.get_angular_velocity()
            ego_acc = ego.get_acceleration()
            ego_control = ego.get_control()

            state = [ego_transform.location.x, ego_transform.location.y, ego_transform.location.z,
                     ego_transform.rotation.pitch, ego_transform.rotation.yaw, ego_transform.rotation.roll,
                     ego_velocity.x, ego_velocity.y, ego_velocity.z,
                     ego_angular.x, ego_angular.y, ego_angular.z,
                     ego_acc.x, ego_acc.y, ego_acc.z,
                     ego_control.throttle, ego_control.steer, ego_control.brake]
            return state

        def get_obstacles_state(ego, obstacles):
            """只记录位置在ego之前，并且距离ego最近的，左右两车道各一个障碍物的位置及大小"""
            ego_location = ego.get_location()
            left_obstacle = None
            right_obstacle = None
            for obstacle in obstacles:
                obstacle_location = obstacle.get_location()
                if obstacle_location.x - ego_location.x > -2.:    ##若ego没有超过障碍物2.m以上, 则需要考虑其影响
                    if not left_obstacle or not right_obstacle:
                        if abs(obstacle_location.y - 204.) <= abs(obstacle_location.y - 207.):
                            left_obstacle = obstacle
                        else:
                            right_obstacle = obstacle
                    else:
                        left_obstacle_location = left_obstacle.get_location()
                        right_obstacle_location = right_obstacle.get_location()
                        obstacle2ego_dist = (ego_location.x - obstacle_location.x)**2 + (ego_location.y - obstacle_location.y)**2
                        if  abs(obstacle_location.y - 204.) <= abs(obstacle_location.y - 207.):
                            current_left2ego_dist = (ego_location.x - left_obstacle_location.x) ** 2 + (ego_location.y - left_obstacle_location.y) ** 2
                            if obstacle2ego_dist <= current_left2ego_dist:
                                left_obstacle = obstacle
                        else:
                            current_right2ego_dist = (ego_location.x - right_obstacle_location.x) ** 2 + (ego_location.y - right_obstacle_location.y) ** 2
                            if obstacle2ego_dist <= current_right2ego_dist:
                                right_obstacle = obstacle

            if not left_obstacle:
                left_obstacle_location = [0., 0., 0.]   ## cause zero make nothing change
            else:
                left_obstacle_location = left_obstacle.get_location()
                left_obstacle_location = [left_obstacle_location.x, left_obstacle_location.y, left_obstacle_location.z]

            if not right_obstacle:
                right_obstacle_location = [0., 0., 0.]  ## cause zero make nothing change
            else:
                right_obstacle_location = right_obstacle.get_location()
                right_obstacle_location = [right_obstacle_location.x, right_obstacle_location.y, right_obstacle_location.z]

            ## [left_pos, left_size, right_pos, right_size]
            state = left_obstacle_location + [3.6, 1.6, 1.5] + right_obstacle_location + [3.6, 1.6, 1.5]
            return state

        def get_lateral_limitation(ego):
            """获取左右可行驶区域的距离"""
            ego_location = ego.get_location()
            left_dist = ego_location.y - env_v1_config.lateral_pos_limitation[0]
            right_dist = env_v1_config.lateral_pos_limitation[1] - ego_location.y
            return [left_dist, right_dist]

        ego_state = get_ego_state(self.ego)
        obstacles_state = get_obstacles_state(self.ego, self.obstacles)
        lateral_state = get_lateral_limitation(self.ego)

        logger.info('ego_state -- ' + str(ego_state))
        logger.info('obstacles_state -- ' + str(obstacles_state))
        logger.info('lateral_state -- ' + str(lateral_state))

        state = ego_state + obstacles_state + lateral_state
        return np.array(state)

    def __respawn_vehicles(self):
        if not env_v1_config.fix_vehicle_pos:
            self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(5, 10))

        obstacles = []
        for idx, vehicle_pos in enumerate(self.vehicles_pos):
            transform = carla.Transform()
            transform.location.x = vehicle_pos[0]
            transform.location.y = vehicle_pos[1]
            transform.location.z = vehicle_pos[2]
            transform.rotation.yaw = -0.142975
            if idx == 0:    ## ego
                self.ego = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                                 role_name='ego', autopilot=False,
                                                                 vehicle_type='vehicle.tesla.model3')
                self.__wait_for_response()
            else:
                obstacle = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                                  role_name='other', autopilot=False)
                self.__wait_for_response()
                obstacles.append(obstacle)
        self.obstacles = obstacles

    def __reattach_sensors(self):
        env_v1_config.collision_sensor_config['attach_to'] = self.ego
        self.collision_sensor = sensor_ops.collision_query(self.world, env_v1_config.collision_sensor_config)

        # env_v1_config.invasion_sensor_config['attach_to'] = self.ego
        # self.lane_invasion_sensor = sensor_ops.lane_invasion_query(self.world, env_v1_config.invasion_sensor_config)
        pass

    def __wait_for_response(self):
        self.world.tick()
        elapse_time = self.world.wait_for_tick()
        # logger.info('respond time consumption %f'%(round(ts.delta_seconds, 6)))
        return elapse_time.delta_seconds

    def __lane_invasion(self):
        """imitate the lane invasion sensor"""
        ego_pos = self.ego.get_location()
        if ego_pos.y > env_v1_config.lateral_pos_limitation[1] or ego_pos.y < env_v1_config.lateral_pos_limitation[0]:
            return True
        else:
            return False

    def __print_pos(self):
        logger.info(str(self.ego.get_location()))

    def __is_done(self):
        """query whether the game done"""
        return self.__lane_invasion() or self.collision_sensor.get()

    def random_action_test_v1(self):
        """ test script, non-syntronic mode
        Example:
            scenario =ObstacleAvoidanceScenario()
            scenario.reset()
            while True:
                state, done = scenario.random_action_test()
                if done:
                    scenario.reset()
                    continue
        """
        state = {}
        done = False
        self.ego.apply_control(carla.VehicleControl(throttle=random.uniform(0., 1.),
                                                    steer=random.uniform(-1., 1.), brake=0.))
        time.sleep(0.5)
        done = self.__is_done()

        return state, done

    def random_action_test_v2(self):
        """ test script, syntronic mode
        Example:
            scenario =ObstacleAvoidanceScenario()
            scenario.reset()
            while True:
                state, done = scenario.random_action_test()
                if done:
                    scenario.reset()
                    continue
        """
        state = None
        done = False

        # --- do action ---#
        self.ego.apply_control(carla.VehicleControl(throttle=0.1,
                                                    steer=0., brake=0.))
        action_time_elapse = 0.
        while True:
            action_time_elapse += self.__wait_for_response()
            if action_time_elapse >= env_v1_config.action_holding_time:
                break

        # ---- get state ---- #
        state = self.__get_env_state()
        done = self.__is_done()

        return state, done

if __name__ == '__main__':
    scenario =ObstacleAvoidanceScenario()
    scenario.reset()
    while True:
        state, done = scenario.random_action_test_v2()
        if done:
            scenario.reset()
            continue
