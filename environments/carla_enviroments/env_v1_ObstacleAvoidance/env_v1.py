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
import os, random
import numpy as np
import math
from gym import spaces
import time as sys_time

class ObstacleAvoidanceScenario(carla_base):
    def __init__(self):
        try:
            carla_base.__init__(self)
        except:
            raise RuntimeError('carla_base init fails...')
        else:
            if env_v1_config.fix_vehicle_pos:
                ## using predefine vehicles position
                self.vehicles_pos = np.load(env_v1_config.vehicles_pos_file)['pos']
            else:
                ## random generate vehicles position
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(5, 10))

            if env_v1_config.synchronous_mode:
                self.start_synchronous_mode()

            # ------ 设置动作空间大小 ----------- #
            self.action_space = spaces.Discrete(len(list(env_v1_config.actions.keys())))

            # ----
            self.max_step = 1000
            self.step_counter = 0

            # end_point
            self.end_point_x = 155.

            # max velocity
            self.max_longitude_velocity = 6. ## m/s
            self.max_lateral_velocity = 1.5 ## m/s


    def reset(self):
        """reset the world"""
        self.__wait_env_running(time=0.5)
        world_ops.destroy_all_actors(self.world)
        self.__wait_env_running(time=0.5)
        self.__respawn_vehicles()
        self.__wait_env_running(time=0.5)
        self.reattach_sensors()

        ## Waiting for the vehicles to land on the ground
        self.__wait_env_running(time=0.5)

        # -- reset some var -- #
        self.last_forward_distance = 0.  ## 用以记录上一次状态时的前行距离
        self.__ego_init_forward_pos = self.ego.get_location().x
        self.__totoal_lane_distance = abs(self.end_point_x - self.__ego_init_forward_pos)

        # step
        self.step_counter = 0
        return self.get_env_state()   # return the init state

    def step(self, action):
        """conduct action in env
        Args:
            action: int, an idx
        Return: env state, reward, done, info
        """
        # --- conduct action and holding a while--- #
        action = env_v1_config.actions[action]
        self.ego.apply_control(carla.VehicleControl(throttle=action[0],
                                                    steer=action[1], brake=action[2]))
        self.__wait_env_running(time=env_v1_config.action_holding_time)

        # -- next state -- #
        state = self.get_env_state()

        # --- reward --- # forward distance, velocity and center pos
        # forward_distance = state[0]
        # velocity = math.sqrt(state[3]**2 + state[4]**2 + 1e-8)
        # lateral_pos = state[1]
        # reward = self.__get_reward_v1(forward_distance=forward_distance, velocity=velocity,
        #                               lateral_pos=lateral_pos)
        reward = self.__get_reward_v1()
        # --reset some var -- #
        # self.last_forward_distance = forward_distance

        self.step_counter += 1

        done = self.__is_done()
        return state, reward, done, {}


    def __get_reward_v1(self, **states):
        # reward = 0.
        #
        # # -- illegal end -- #
        # if self.__is_illegal_done():
        #     reward -= 3.
        #
        # # -- forward distance -- #
        # r_f_cur = states['forward_distance'] * 10.
        # r_f_last = self.last_forward_distance * 10.
        # r_f = r_f_cur - r_f_last
        # # print('foward_reward:' + str(r_f), 'foward_distance:', states['forward_distance'],
        # #       'last_forward_distance:', self.last_forward_distance)
        # reward += r_f
        # # print('forward_reward:', r_f)
        #
        # # --- velocity --- #
        # r_v = self.__gaussian_1d(x=states['velocity'], mean=6., std=4., max=2., bias=0.) - 0.1
        # # logger.info('r_v:' + str(r_v))
        # reward += r_v
        # # print('velocity_reward:', r_v, ',  total_reward:', reward)
        # # -- lateral center -- #
        # todo
        return -2. if self.__is_illegal_done() else 1.

    def __gaussian_1d(self, x, mean, std, max, bias):
        def norm(x, mu, sigma):
            """normal gaussian function
            """
            # print(sigma)
            mu = np.array(mu)
            sigma = np.array(sigma)
            x = np.array(x)
            pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
            return pdf

        pdf = norm(x=x, mu=mean, sigma=std) - norm(x=bias, mu=mean, sigma=std)  ## raw gaussian - bias
        pdf = pdf / (norm(x=mean, mu=mean, sigma=std)- norm(x=bias, mu=mean, sigma=std)) * max
        return pdf

    def get_env_state(self):
        def get_ego_state(ego):
            ego_transform = ego.get_transform()
            ego_velocity = ego.get_velocity()
            ego_angular = ego.get_angular_velocity()
            ego_acc = ego.get_acceleration()
            ego_control = ego.get_control()

            # print('ego_transform.rotation.yaw', ego_transform.rotation.yaw)
            # print('ego_transform.rotation.pitch', ego_transform.rotation.pitch)
            # print('ego_transform.rotation.roll', ego_transform.rotation.roll)
            # print('ego_angular.x:', ego_angular.x)
            # print('ego_angular.y:', ego_angular.y)
            # print('ego_angular.z:', ego_angular.z)
            # print('x_v', ego_velocity.x)
            # print('y_v', ego_velocity.y)
            lateral = abs(env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0])
            init_lateral_point = (env_v1_config.lateral_pos_limitation[1] + env_v1_config.lateral_pos_limitation[0]) / 2.
            haft_lateral = lateral / 2.
            # state = [(ego_transform.location.x - self.__ego_init_forward_pos)/self.__totoal_lane_distance,
            #          (ego_transform.location.y - env_v1_config.lateral_pos_limitation[0])/lateral,
            #          ego_transform.rotation.yaw / 30.,
            #          ego_angular.z / 50.,
            #          ego_velocity.x / self.max_longitude_velocity,
            #          ego_velocity.y / self.max_lateral_velocity]
            #          # ego_acc.x, ego_acc.y,
            #          # ego_control.throttle, ego_control.steer, ego_control.brake]

            state = [(ego_transform.location.y - init_lateral_point) / haft_lateral,
                     ego_velocity.y / self.max_lateral_velocity,
                     ego_transform.rotation.yaw / 30.,
                     ego_angular.z / 50.]

            # state[0] = max(state[0], 1e-2) if state[0] >=0 else min(state[0], -1e-2)
            # state[1] = max(state[1], 1e-2) if state[1] >=0 else min(state[1], -1e-2)
            # state[2] = max(state[2], 1e-2) if state[2] >=0 else min(state[2], -1e-2)
            # state[3] = max(state[3], 1e-2) if state[3] >=0 else min(state[3], -1e-2)


            # ego_acc.x, ego_acc.y,
            # ego_control.throttle, ego_control.steer, ego_control.brake]
            # print('state:', state)
            return state

        def get_obstacles_state(ego, obstacles):
            """只记录位置在ego之前，并且距离ego最近的，左右两车道各一个障碍物的位置及大小"""
            # ---- 获取左右车道线离ego最近的车辆（考虑范围为没有超过障碍物3.8m以上的所有） ----- #
            ego_location = ego.get_location()
            left_obstacle = None
            right_obstacle = None
            for obstacle in obstacles:
                obstacle_location = obstacle.get_location()
                if obstacle_location.x - ego_location.x > -3.8:    ##若ego没有超过障碍物3.8m以上, 则需要考虑其影响
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
            # ---- 获取左右车道线离ego最近的车辆（考虑范围为没有超过障碍物3.8m以上的所有） ----- #

            lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]

            # --- 若没有，则默认0 ---- #
            if not left_obstacle:
                left_obstacle_location = [0., 0., 0.]   ##
            else:
                left_obstacle_location = left_obstacle.get_location()
                left_obstacle_location = [1., ## 指示有没有左障碍物
                                          (left_obstacle_location.x - ego_location.x) / self.__totoal_lane_distance,
                                          (left_obstacle_location.y - ego_location.y)/lateral]

            if not right_obstacle:
                right_obstacle_location = [0., 0., 0.]  ##
            else:
                right_obstacle_location = right_obstacle.get_location()
                right_obstacle_location = [1.,
                                           (right_obstacle_location.x - ego_location.x) / self.__totoal_lane_distance,
                                           (right_obstacle_location.y - ego_location.y)/lateral]

            ## [left_pos, left_size, right_pos, right_size]
            state = left_obstacle_location + right_obstacle_location
            return state

        def get_lateral_limitation(ego):
            """获取左右可行驶区域的距离"""
            ego_location = ego.get_location()
            lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]
            left_dist = (ego_location.y - env_v1_config.lateral_pos_limitation[0]) / lateral
            right_dist = (env_v1_config.lateral_pos_limitation[1] - ego_location.y) / lateral
            return [left_dist, right_dist]

        ego_state = get_ego_state(self.ego)
        obstacles_state = get_obstacles_state(self.ego, self.obstacles)
        lateral_state = get_lateral_limitation(self.ego)

        # logger.info('ego_state -- ' + str(ego_state))
        # logger.info('obstacles_state -- ' + str(obstacles_state))
        # logger.info('lateral_state -- ' + str(lateral_state))

        state = ego_state + obstacles_state
        # state = ego_state + lateral_state
        # state = ego_state   ## this is ok for only balance driving
        return np.array(state)

    def __respawn_vehicles(self):
        only_one_vehicle = False

        if not only_one_vehicle:
            if not env_v1_config.fix_vehicle_pos:
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(8, 12))

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
                    self.wait_for_response()
                    pass
                else:
                    obstacle = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                                      role_name='other', autopilot=False)
                    self.wait_for_response()
                    obstacles.append(obstacle)
            self.obstacles = obstacles
        else:
            vehicle_pos = [-30., (env_v1_config.lateral_pos_limitation[0] + env_v1_config.lateral_pos_limitation[1])/2., 1.81]
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


    def reattach_sensors(self):
        env_v1_config.collision_sensor_config['attach_to'] = self.ego
        self.collision_sensor = sensor_ops.collision_query(self.world, env_v1_config.collision_sensor_config)

        # env_v1_config.invasion_sensor_config['attach_to'] = self.ego
        # self.lane_invasion_sensor = sensor_ops.lane_invasion_query(self.world, env_v1_config.invasion_sensor_config)
        pass

    def __lane_invasion(self):
        """imitate the lane invasion sensor"""
        ego_pos = self.ego.get_location()
        if ego_pos.y > env_v1_config.lateral_pos_limitation[1] or ego_pos.y < env_v1_config.lateral_pos_limitation[0]:
            return True
        else:
            return False


    def __print_pos(self):
        # logger.info(str(self.ego.get_location()))
        pass


    def __is_finish_game(self):
        """judge agent whether finish game"""
        location = self.ego.get_location()
        return True if location.x >= self.end_point_x else False


    def __is_illegal_done(self):
        """ judge whether lane invasion or collision"""
        return self.__lane_invasion() or self.collision_sensor.get()


    def __is_done(self):
        """query whether the game done"""
        max_step = self.step_counter >= self.max_step
        return self.__is_finish_game() or self.__lane_invasion() or self.collision_sensor.get() or max_step


    def __wait_env_running(self, time):
        """等待环境跑一段时间
        Args:
            time: sec
        """
        if env_v1_config.synchronous_mode:
            self.wait_carla_runing(time)
        else:
            sys_time.sleep(time)


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
        sys_time.sleep(0.5)
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
        steer = random.uniform(-1., 1.)
        self.ego.apply_control(carla.VehicleControl(throttle=0.5,
                                                    steer=0., brake=0.))
        # ---- action holding ---- #
        self.__wait_env_running(time=env_v1_config.action_holding_time)

        # ---- get state ---- #
        state = self.get_env_state()

        # --- reward --- # forward distance, velocity and center pos
        reward = self.__get_reward_v1()
        logger.info('reward - %f'%(reward))

        done = self.__is_done()
        return state, done

    def conduct(self):
        pass

if __name__ == '__main__':
    scenario =ObstacleAvoidanceScenario()
    scenario.reset()
    while True:
        state, done = scenario.random_action_test_v2()
        if done:
            scenario.reset()
            continue
