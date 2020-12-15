import copy
import torch
import torch.nn as nn

from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DQN import DQN

from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer
import torch.optim as optim

import os
from environments.carla_enviroments.carla_config import base_config

import time

from agents.DQN_agents.base_conv_net.mobilenet_v2 import MobileNetV2
from nn_builder.pytorch.NN import NN

from agents.DQN_agents.q_networks.two_eye_q_network import TwoEyesQNetwork
from agents.DQN_agents.q_networks.single_eye_q_network import SingleEyeQNetwork

class DQN_With_Fixed_Q_Targets(DQN):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"
    def __init__(self, config):
        DQN.__init__(self, config)
        self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

        if config.resume:
            self.load_resume(config.resume_path)

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        super(DQN_With_Fixed_Q_Targets, self).learn(experiences=experiences)

        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])  # Update the target network
        # print('learn time:%.5f, soft copy:%.5f'%(tic2 - tic1, tic3 - tic2))

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def load_resume(self, resume_path):
        save = torch.load(resume_path)
        q_network_local_dict = save['q_network_local']
        q_network_target_dict = save['q_network_target']
        self.q_network_local.load_state_dict(q_network_local_dict, strict=True)
        self.q_network_target.load_state_dict(q_network_target_dict, strict=True)
        self.logger.info('load resume model success...')

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        # if self.episode_number >= self.total_episode * 3 / 4:
        #     new_lr = starting_lr / 6.
        # elif self.episode_number >= self.total_episode / 2:
        #     new_lr = starting_lr / 3.
        # else:
        #     new_lr = starting_lr

        new_lr = starting_lr

        for g in optimizer.param_groups:
            g['lr'] = new_lr

        self.logger.info("Learning rate {}".format(new_lr))


class DQN_With_Fixed_Q_Targets_SINGLE_EYE(DQN_With_Fixed_Q_Targets):
    agent_name = "DQN_SINGLE_EYE"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        base_config.no_render_mode = False  ## must be render mode

        self.q_network_local = SingleEyeQNetwork(n_action=self.get_action_size(), pretrain=config.backbone_pretrain,
                                                 bn=self.hyperparameters["batch_norm"])
        self.q_network_target = SingleEyeQNetwork(n_action=self.get_action_size(), pretrain=False,
                                                  bn=self.hyperparameters["batch_norm"])
        self.q_network_optimizer = optim.SGD(self.q_network_local.parameters(),
                                             lr=self.hyperparameters["learning_rate"], weight_decay=5e-4)

        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                                    config.seed)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

        self.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

        self.q_network_local.to(self.q_network_local.device)
        self.q_network_target.to(self.q_network_target.device)

        if config.resume:
            self.load_resume(config.resume_path)


class DQN_With_Fixed_Q_Targets_2_EYE(DQN_With_Fixed_Q_Targets):

    agent_name = "DQN_TWO_EYES"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        base_config.no_render_mode = False  ## must be render mode

        self.q_network_local = TwoEyesQNetwork(n_action=self.get_action_size())
        self.q_network_target = TwoEyesQNetwork(n_action=self.get_action_size())
        self.q_network_optimizer = optim.SGD(self.q_network_local.parameters(),
                                             lr=self.hyperparameters["learning_rate"], weight_decay=5e-4)

        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                                    config.seed)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)


        self.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

        self.q_network_local.to(self.q_network_local.device)
        self.q_network_target.to(self.q_network_target.device)


    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if self.episode_number >= self.total_episode / 3:
            new_lr = starting_lr / 10.
        elif self.episode_number >= self.total_episode * 2 / 3:
            new_lr = starting_lr / 100.
        else:
            new_lr = starting_lr

        for g in optimizer.param_groups:
            g['lr'] = new_lr

        self.logger.info("Learning rate {}".format(new_lr))


if __name__ == '__main__':
    from utilities.data_structures.Config import Config
    import gym
    ## envs import ##
    from environments.carla_enviroments import env_v1_ObstacleAvoidance

    # net = q_network_toa(n_action=4)
    # net.to('cuda')
    # input = torch.rand(size=(10, 3, 224, 224)).to('cuda')
    # q1, q2 = net(input)

    config = Config()
    config.seed = 1
    config.environment = gym.make("ObstacleAvoidance-v0")
    config.num_episodes_to_run = 2000
    config.file_to_save_data_results = "C:/my_project/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/results/data_and_graphs/carla_obstacle_avoidance/data.pkl"
    config.file_to_save_results_graph = "C:/my_project/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/results/data_and_graphs/carla_obstacle_avoidance/data.png"
    config.show_solution_score = False
    config.visualise_individual_results = True
    config.visualise_overall_agent_results = True
    config.standard_deviation_results = 1.0
    config.runs_per_agent = 1
    config.use_GPU = True
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = True

    config.resume = False
    config.resume_path = ''
    config.backbone_pretrain = True

    config.hyperparameters = {
            "learning_rate": 1e-2 * 10.,
            "batch_size": 32,
            "buffer_size": 20000,
            "epsilon": 1.0,
            "epsilon_decay_rate_denominator": 1.0,
            "discount_rate": 0.99,
            "tau": 0.01,
            "alpha_prioritised_replay": 0.6,
            "beta_prioritised_replay": 0.1,
            "incremental_td_error": 1e-8,
            "update_every_n_steps": 1,
            "linear_hidden_units": [24, 48, 24],
            "final_layer_activation": "None",
            "batch_norm": False,
            "gradient_clipping_norm": 0.1,
            "learning_iterations": 1,
            "clip_rewards": False
    }

    dqn_net = DQN_With_Fixed_Q_Targets_2_EYE(config)
    # left_input = torch.rand(size=(5, 3, 224, 224)).to('cuda')
    # right_input = torch.rand(size=(5, 3, 224, 224)).to('cuda')
    # out1 = dqn_net.q_network_local(left_input, right_input)
    # out2 = dqn_net.q_network_target(left_input, right_input)
    pass


