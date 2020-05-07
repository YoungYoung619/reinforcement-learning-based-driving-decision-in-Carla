import gym
from gym import spaces
import torch

#--------- 这句import一定要加---------#
from environments.carla_enviroments import env_v1_ObstacleAvoidance

if __name__ == '__main__':
    env = gym.make('ObstacleAvoidance-v0')

    env.reset()
    while True:
        state, done = env.random_action_test_v2()
        if done:
            env.reset()
            continue
    pass

