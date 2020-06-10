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
import cv2

from environments.carla_enviroments.utils.kp2hm import heat_map
from environments.carla_enviroments.utils.kp2hm import gaussian_1d
from environments.carla_enviroments.utils.kp2hm import gaussian_2d

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
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(5, 15))

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
            self.max_longitude_velocity = 6. ## m/s, this is not the speed limitation, only used to normalize the velocity
            self.max_lateral_velocity = 1.5 ## m/s, same as above

            ##
            self.n_been_catched_record = 0

            self.test_mode = False

            self.left_obstacle = None
            self.right_obstacle = None


    def reset(self):
        """reset the world"""
        self.__wait_env_running(time=0.5)
        world_ops.destroy_all_actors(self.world)
        self.__wait_env_running(time=0.5)
        self.respawn_vehicles()
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
        self.n_been_catched_record = 0
        return self.get_env_state()   # return the init state

    def step(self, action_idx):
        """conduct action in env
        Args:
            action: int, an idx
        Return: env state, reward, done, info
        """
        # --- conduct action and holding a while--- #
        if self.test_mode:
            action = np.array(env_v1_config.actions[action_idx])
            action = 0.7 * action + 0.3 * self.last_action
            self.last_action = action
        else:
            action = env_v1_config.actions[action_idx]

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
        # reward = self.__get_reward_v1()
        reward = self.__get_reward_v3(state=state)
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

    def __get_reward_v2(self, **states):
        reward = 0.
        if self.__get_n_been_catched_so_far() > self.n_been_catched_record:
            # print('catch one')
            reward = 1.
            self.n_been_catched_record = self.__get_n_been_catched_so_far()
        return -1. if self.__is_illegal_done() else reward

    def __get_reward_v3(self, **states):
        obj_sigma = 10
        lane_sigma = 5
        state = states['state']

        lateral_pos = state[0]  ## [-1, 1]
        lateral_pos_x_pixel = int(lateral_pos * 100)
        lateral_pos_y_pixel = 512 // 2

        left_line_point = (lateral_pos_y_pixel - 100, 0)
        right_line_point = (lateral_pos_y_pixel + 100, 0)
        lane_reward_map = heat_map((512, 512),
                                  [left_line_point, right_line_point],
                                  sigma=lane_sigma, func=gaussian_1d)

        left_obstacle = state[4:7]
        if left_obstacle[0]:
            lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]
            left_pos_x_pixel = int(left_obstacle[2] * 2 * 100 + 256 + lateral_pos_x_pixel)
            left_pos_y_pixel = int(256 - (left_obstacle[1] * self.__totoal_lane_distance / 150 * 256))

        right_obstacle = state[7:10]
        if right_obstacle[0]:
            right_pos_x_pixel = int(right_obstacle[2] * 2 * 100 + 256 + lateral_pos_x_pixel)
            right_pos_y_pixel = int(256 - (right_obstacle[1] * self.__totoal_lane_distance / 150 * 256))

        if left_obstacle[0] and right_obstacle[0]:
            obstacle_reward_map = heat_map((512, 512),
                                  [(left_pos_x_pixel, left_pos_y_pixel), (right_pos_x_pixel, right_pos_y_pixel)],
                                  sigma=obj_sigma, func=gaussian_2d)
        elif left_obstacle[0] and not bool(right_obstacle[0]):
            obstacle_reward_map = heat_map((512, 512),
                                  [(left_pos_x_pixel, left_pos_y_pixel)],
                                  sigma=obj_sigma, func=gaussian_2d)
        elif not bool(left_obstacle[0]) and right_obstacle[0]:
            obstacle_reward_map = heat_map((512, 512),
                                  [(right_pos_x_pixel, right_pos_y_pixel)],
                                  sigma=obj_sigma, func=gaussian_2d)
        else:
            obstacle_reward_map = np.zeros(shape=(512, 512), dtype=np.float32)

        reward_obstacle = - obstacle_reward_map[(lateral_pos_y_pixel, 256 + lateral_pos_x_pixel)] * 4.
        reward_lane = - lane_reward_map[(lateral_pos_y_pixel, 256 + lateral_pos_x_pixel)] * 3.

        reward_exist = 0.
        if not self.__is_illegal_done():
            reward_exist = 1.

        reward = reward_exist + reward_obstacle + reward_lane
        # print('reward:', reward_lane)
        # cv2.imshow('test_r', np.maximum(obstacle_reward_map, lane_reward_map))
        return reward

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

    def get_ego_state(self, ego):
        ego_transform = ego.get_transform()
        ego_velocity = ego.get_velocity()
        ego_angular = ego.get_angular_velocity()
        # ego_acc = ego.get_acceleration()
        # ego_control = ego.get_control()

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

    def get_obstacles_state(self, ego, obstacles):
        """只记录位置在ego之前，并且距离ego最近的，左右两车道各一个障碍物的位置及大小"""
        # ---- 获取左右车道线离ego最近的车辆（考虑范围为没有超过障碍物3.8m以上的所有） ----- #
        ego_location = ego.get_location()
        self.right_obstacle = None
        self.left_obstacle = None
        for obstacle in obstacles:
            obstacle_location = obstacle.get_location()
            if obstacle_location.x - ego_location.x > -3.8:  ##若ego没有超过障碍物3.8m以上, 则需要考虑其影响
                if not self.left_obstacle:
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) <= abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        self.left_obstacle = obstacle
                else:
                    left_obstacle_location = self.left_obstacle.get_location()
                    obstacle2ego_dist = (ego_location.x - obstacle_location.x) ** 2 + (
                            ego_location.y - obstacle_location.y) ** 2
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) <= abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        current_left2ego_dist = (ego_location.x - left_obstacle_location.x) ** 2 + (
                                    ego_location.y - left_obstacle_location.y) ** 2
                        if obstacle2ego_dist <= current_left2ego_dist:
                            self.left_obstacle = obstacle

                if not self.right_obstacle:
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) > abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        self.right_obstacle = obstacle
                else:
                    right_obstacle_location = self.right_obstacle.get_location()
                    obstacle2ego_dist = (ego_location.x - obstacle_location.x) ** 2 + (
                            ego_location.y - obstacle_location.y) ** 2
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) > abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        current_right2ego_dist = (ego_location.x - right_obstacle_location.x) ** 2 + (
                                ego_location.y - right_obstacle_location.y) ** 2
                        if obstacle2ego_dist <= current_right2ego_dist:
                            self.right_obstacle = obstacle
        # ---- 获取左右车道线离ego最近的车辆（考虑范围为没有超过障碍物3.8m以上的所有） ----- #

        lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]

        # --- 若没有，则默认0 ---- #
        if not self.left_obstacle:
            left_obstacle_location = [0., 0., 0.]  ##
        else:
            left_obstacle_location = self.left_obstacle.get_location()
            if math.sqrt((left_obstacle_location.x - ego_location.x) ** 2 + (
                    left_obstacle_location.y - ego_location.y) ** 2) <= env_v1_config.farest_vehicle_consider:
                left_obstacle_location = [1.,  ## 指示有没有左障碍物
                                          (left_obstacle_location.x - ego_location.x) / self.__totoal_lane_distance,
                                          (left_obstacle_location.y - ego_location.y) / lateral]
            else:
                left_obstacle_location = [0., 0., 0.]  ##

        if not self.right_obstacle:
            right_obstacle_location = [0., 0., 0.]  ##
        else:
            right_obstacle_location = self.right_obstacle.get_location()
            if math.sqrt((right_obstacle_location.x - ego_location.x) ** 2 + (
                    right_obstacle_location.y - ego_location.y) ** 2) <= env_v1_config.farest_vehicle_consider:
                right_obstacle_location = [1.,
                                           (right_obstacle_location.x - ego_location.x) / self.__totoal_lane_distance,
                                           (right_obstacle_location.y - ego_location.y) / lateral]
            else:
                right_obstacle_location = [0., 0., 0.]  ## 右边的最近车辆太远，当做没有

        ## [left_pos, left_size, right_pos, right_size]
        state = left_obstacle_location + right_obstacle_location
        return state

    def get_lateral_limitation(self, ego):
        """获取左右可行驶区域的距离"""
        ego_location = ego.get_location()
        lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]
        left_dist = (ego_location.y - env_v1_config.lateral_pos_limitation[0]) / lateral
        right_dist = (env_v1_config.lateral_pos_limitation[1] - ego_location.y) / lateral
        return [left_dist, right_dist]

    def get_env_state(self):
        ego_state = self.get_ego_state(self.ego)
        obstacles_state = self.get_obstacles_state(self.ego, self.obstacles)
        # lateral_state = self.get_lateral_limitation(self.ego)

        # logger.info('ego_state -- ' + str(ego_state))
        # logger.info('obstacles_state -- ' + str(obstacles_state))
        # logger.info('lateral_state -- ' + str(lateral_state))

        state = ego_state + obstacles_state
        # state = ego_state + lateral_state
        # state = ego_state   ## this is ok for only balance driving
        return np.array(state)

    def respawn_vehicles(self):
        only_one_vehicle = False

        if not only_one_vehicle:
            if not env_v1_config.fix_vehicle_pos:
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(12, 12))

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

        if self.test_mode:
            self.attach_camera_to_ego()

        # env_v1_config.invasion_sensor_config['attach_to'] = self.ego
        # self.lane_invasion_sensor = sensor_ops.lane_invasion_query(self.world, env_v1_config.invasion_sensor_config)

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
        # self.ego.apply_control(carla.VehicleControl(throttle=0.7,
        #                                             steer=0., brake=0.))
        self.step(action_idx=4)

        # ---- action holding ---- #
        self.__wait_env_running(time=env_v1_config.action_holding_time)

        # ---- get state ---- #
        state = self.get_env_state()
        # ego_state = state[:4]
        # left_obstacle = state[4:7]
        # right_obstacle = state[7:]
        # print('ego_state:', ego_state)
        # print('left_obstacle:', left_obstacle)
        # print('right_obstacle:', right_obstacle)


        # --- reward --- # forward distance, velocity and center pos
        # reward = self.__get_reward_v2()
        reward = self.__get_reward_v3(state=state)
        # logger.info('reward - %f'%(reward))

        done = self.__is_done()

        self.vis_state_as_img(state)
        return state, done

    def vis_state_as_img(self, state):
        img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

        lateral_pos = state[0] ## [-1, 1]
        lateral_pos_x_pixel = int(lateral_pos * 100)
        lateral_pos_y_pixel = 512 // 2

        ## draw lane
        cv2.line(img, (lateral_pos_y_pixel-100, 0), (lateral_pos_y_pixel-100, 512),
                 color=(0, 255, 0), thickness=2)

        cv2.line(img, (lateral_pos_y_pixel + 100, 0), (lateral_pos_y_pixel + 100, 512),
                 color=(0, 255, 0), thickness=2)


        ## draw ego vehicle
        cv2.circle(img, (256 + lateral_pos_x_pixel, lateral_pos_y_pixel), color=(255, 0, 0),
                   thickness=3, radius=10)

        ##
        left_obstacle = state[4:7]
        if left_obstacle[0]:
            lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]
            left_pos_x_pixel = int(left_obstacle[2] * 2 * 100 + 256 + lateral_pos_x_pixel)
            left_pos_y_pixel = int(256 - (left_obstacle[1] * self.__totoal_lane_distance / 150 * 256))
            cv2.circle(img, (left_pos_x_pixel, left_pos_y_pixel), color=(0, 0, 255),
                       thickness=3, radius=10)

        right_obstacle = state[7:10]
        if right_obstacle[0]:
            right_pos_x_pixel = int(right_obstacle[2] * 2 * 100 + 256 + lateral_pos_x_pixel)
            right_pos_y_pixel = int(256 - (right_obstacle[1] * self.__totoal_lane_distance / 150 * 256))

            cv2.circle(img, (right_pos_x_pixel, right_pos_y_pixel), color=(0, 0, 255),
                       thickness=3, radius=10)

        if left_obstacle[0] and right_obstacle[0]:
            reward_map = heat_map((512, 512), [(left_pos_x_pixel, left_pos_y_pixel), (right_pos_x_pixel, right_pos_y_pixel)], sigma=20, func=gaussian_2d)
        elif left_obstacle[0] and not bool(right_obstacle[0]):
            reward_map = heat_map((512, 512),
                                  [(left_pos_x_pixel, left_pos_y_pixel)],
                                  sigma=50, func=gaussian_2d)
        elif not bool(left_obstacle[0]) and right_obstacle[0]:
            reward_map = heat_map((512, 512),
                                  [(right_pos_x_pixel, right_pos_y_pixel)],
                                  sigma=50, func=gaussian_2d)
        else:
            reward_map = np.zeros(shape=(512, 512), dtype=np.uint8)

        cv2.imshow('reward', reward_map)

        cv2.imshow('test', img)
        cv2.waitKey(1)

    def random_action_test_v3(self, throttle, steer):
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
        self.ego.apply_control(carla.VehicleControl(throttle=throttle,
                                                    steer=steer, brake=0.))
        # ---- action holding ---- #
        self.__wait_env_running(time=env_v1_config.action_holding_time)

        # ---- get state ---- #
        state = self.get_env_state()

        # --- reward --- # forward distance, velocity and center pos
        reward = self.__get_reward_v1()
        logger.info('reward - %f'%(reward))

        done = self.__is_done()
        return state, done

    def __get_n_been_catched_so_far(self):
        n_been_catched_so_far = 0
        ego_pos = self.ego.get_location()

        for obstacle in self.obstacles:
            obstacle_pos = obstacle.get_location()
            if ego_pos.x - obstacle_pos.x > 0.1:
                n_been_catched_so_far += 1
        return n_been_catched_so_far

    def attach_camera_to_ego(self):
        camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': 640,
                              'image_size_y': 360, 'fov': 110, 'sensor_tick': 0.02,
                              'transform': carla.Transform(carla.Location(x=-0., y=-0.4, z=1.25)),
                              'attach_to': self.ego}

        self.camera = sensor_ops.bgr_camera(self.world, camera_config)

    def test(self):
        self.last_action = np.array([0., 0., 0.])
        self.test_mode = True

if __name__ == '__main__':

    scenario =ObstacleAvoidanceScenario()
    scenario.reset()
    t = True
    while True:
        _, done = scenario.random_action_test_v2()

        if done:
            scenario.reset()
            continue
