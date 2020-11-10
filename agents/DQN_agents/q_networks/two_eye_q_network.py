"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
from torch import nn
import torch
import os
from agents.DQN_agents.base_conv_net.mobilenet_v2 import MobileNetV2
from nn_builder.pytorch.NN import NN

class TwoEyesQNetwork(nn.Module):
    def __init__(self, n_action, pretrain=True, pretrain_file=os.path.join(os.path.dirname(__file__),
                                                                           '../base_conv_net/pretrain/mobilenetv2_0.35-b2e15951.pth')):
        super(TwoEyesQNetwork, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.backbone = MobileNetV2(width_mult=0.35, n_dim=128)

        ## action layer
        action_layer_config = {"linear_hidden_units": [128, 64, 32],
                                  "final_layer_activation": "None",
                                  "batch_norm": False}
        self.action_layer = self.create_NN(256, n_action, hyperparameters=action_layer_config)

        if pretrain:
            self.load_pretrain(pretrain_file)

    def forward(self, state):
        left_eye = state[..., :3].transpose(1, 3).transpose(2, 3).contiguous()
        right_eye = state[..., 3:].transpose(1, 3).transpose(2, 3).contiguous()
        features_left = self.backbone(left_eye) ## shape is [bs, 256]
        features_right = self.backbone(right_eye)  ## shape is [bs, 256]
        features = torch.cat([features_left, features_right], dim=-1)    ## shape is [bs, 512]
        action_q = self.action_layer(features)
        # print('transpose:%.5fs, cnn:%.5fs, bp:%.5f'%(tic2 - tic1, tic3 - tic2, tic4 - tic3))
        return action_q

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

    def load_pretrain(self, pretrain_model_path):
        """load pretrain model"""
        net_dict = self.backbone.state_dict()
        if not torch.cuda.is_available():
            pretrain_dict = torch.load(pretrain_model_path, map_location='cpu')
        else:
            pretrain_dict = torch.load(pretrain_model_path)
        # print(net_dict.keys())
        # print(pretrain_dict.keys())

        load_dict = {(k): v for k, v in pretrain_dict.items() if
                     (k) in net_dict}
        # print(load_dict.keys())

        # last_conv_layer_net_dict_name = ['features.17.0.weight', 'features.17.1.weight', 'features.17.1.bias',
        #                                  'features.17.1.running_mean', 'features.17.1.running_var', 'features.17.1.num_batches_tracked']
        # last_conv_layer_pretrain_dict_name = ['conv.0.weight', 'conv.1.weight', 'conv.1.bias', 'conv.1.running_mean',
        #                                       'conv.1.running_var', 'conv.1.num_batches_tracked']
        #
        # for net_dict_key, pretrain_dict_key in zip(last_conv_layer_net_dict_name, last_conv_layer_pretrain_dict_name):
        #     if net_dict[net_dict_key].size() == pretrain_dict[pretrain_dict_key].size():
        #         load_dict[net_dict_key] = pretrain_dict[pretrain_dict_key]
        #         print(net_dict_key, '  ok')
        #     else:
        #         print(net_dict[net_dict_key].size(), pretrain_dict[pretrain_dict_key].size())

        net_dict.update(load_dict)
        self.backbone.load_state_dict(net_dict, strict=True)
        print(f'load keys:{load_dict.keys()}')
