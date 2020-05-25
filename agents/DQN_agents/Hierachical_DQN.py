from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.base_conv_net.mobilenet_v2 import MobileNetV2

from nn_builder.pytorch.NN import NN
import torch
import torch.nn as nn
import math
import os
import gym

class q_network_toa(nn.Module):
    def __init__(self, n_action):
        super(q_network_toa, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.backbone = MobileNetV2(width_mult=0.35)

        ## take_over_layer
        take_over_layer_config = {"linear_hidden_units": [256, 128, 64],
                                  "final_layer_activation": "None",
                                  "batch_norm": False}
        self.take_over_layer = self.create_NN(512, 2, hyperparameters=take_over_layer_config)

        ## action layer
        action_layer_config = {"linear_hidden_units": [256, 128, 64],
                                  "final_layer_activation": "None",
                                  "batch_norm": False}
        self.action_layer = self.create_NN(512, n_action, hyperparameters=action_layer_config)

        self._initialize_weights()

    def forward(self, input):
        features = self.backbone(input)
        take_over_q = self.take_over_layer(features)
        action_q = self.action_layer(features)
        return take_over_q, action_q

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use"""
        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "he", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Hierachical_DQN(DQN):

    def __init__(self, config):
        DQN.__init__(self, config)
        self.q_network_local = q_network_toa(n_action=self.get_action_size())
        self.q_network_target = q_network_toa(n_action=self.get_action_size())

        if config.backbone_pretrain:
            self.load_pretrain()

        self.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

        self.q_network_local.to(self.q_network_local.device)
        self.q_network_target.to(self.q_network_target.device)

    def load_pretrain(self):
        pretrain_model_path = os.path.join(os.path.dirname(__file__), 'base_conv_net/pretrain/mobilenetv2_0.35-b2e15951.pth')
        net_dict = self.q_network_local.state_dict()
        if not torch.cuda.is_available():
            pretrain_dict = torch.load(pretrain_model_path, map_location='cpu')
        else:
            pretrain_dict = torch.load(pretrain_model_path)
        # print(net_dict.keys())
        # print(pretrain_dict.keys())

        aa = list(net_dict.keys())
        bb = list(net_dict.values())
        cc = list(pretrain_dict.keys())
        dd = list(pretrain_dict.values())

        load_dict = {('backbone.' + k): v for k, v in pretrain_dict.items() if
                     ('backbone.' + k) in net_dict}
        net_dict.update(load_dict)
        self.q_network_local.load_state_dict(net_dict, strict=True)
        print(f'load keys:{load_dict.keys()}')
        self.logger.info(f'load keys:{load_dict.keys()}')


if __name__ == '__main__':
    from utilities.data_structures.Config import Config
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

    config.resume = True
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

    dqn_net = Hierachical_DQN(config)
    input = torch.rand(size=(5, 3, 224, 224)).to('cuda')
    out1 = dqn_net.q_network_local(input)
    out2 = dqn_net.q_network_target(input)

    pass