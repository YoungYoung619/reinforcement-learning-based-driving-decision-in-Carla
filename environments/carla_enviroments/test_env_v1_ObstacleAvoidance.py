import gym
from gym import spaces
import torch
from nn_builder.pytorch.NN import NN
import numpy as np

import time
from environments.carla_enviroments.carla_config import base_config
from environments.carla_enviroments.env_v1_ObstacleAvoidance import env_v1_config
import cv2

#--------- 这句import一定要加---------#
from environments.carla_enviroments import env_v1_ObstacleAvoidance

hyperparameters = {
        "learning_rate": 1e-2*10.,
        "batch_size": 64,
        "buffer_size": 20000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 10,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [32, 108, 296, 108, 32],
        "final_layer_activation": "None",
        "batch_norm": True,
        "gradient_clipping_norm": 0.1,
        "learning_iterations": 1,
        "clip_rewards": False}

resume_path = 'C:\my_project\\reinforcement-learning-based-driving-decision-in-Carla\\results\Models\ObstacleAvoidance-v2\\rolling_score_39.4644.model'

def create_NN(input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
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
              random_seed=1).to(torch.device('cuda'))

# def get_action_size(env):
#     """Gets the action_size for the gym env into the correct shape for a neural network"""
#
#     self.environment.action_space.n
#     else: return self.environment.action_space.shape[0]

def load_q_network(input_dim, output_dim):
    save = torch.load(resume_path)
    resume_dict = save['q_network_local']
    net = create_NN(input_dim, output_dim, hyperparameters=hyperparameters)
    net.load_state_dict(resume_dict, strict=True)
    net.eval()
    return net

def get_action(net, state):
    state = torch.from_numpy(state).cuda().unsqueeze(dim=0).float()
    with torch.no_grad():
        out = net(state)
        action_idx = out.argmax(dim=-1).item()
    return action_idx

if __name__ == '__main__':
    ## save video
    save_video = True
    save_to = 'DDQN_1111.avi'

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        VideoWriter = cv2.VideoWriter(save_to, fourcc, int(1./0.25), (640, 360))

    base_config.no_render_mode = False
    env = gym.make('ObstacleAvoidance-v2')
    env.test()

    state = env.reset()
    net = load_q_network(state.size, env.action_space.n)
    total_reward = 0.
    while True:
        tic1 = time.time()
        action_idx = get_action(net, state)
        state, reward, done, _ = env.step(action_idx)
        total_reward += reward
        action_hold = time.time() - tic1
        print('action_hold:%.4ssec'%(action_hold))
        img = env.camera.get()

        if save_video:
            VideoWriter.write(img)
            # time.sleep(1. / 25.)
        else:
            cv2.imshow('camera', img)
            cv2.waitKey(1)

        if done:
            state = env.reset()
            print('total_reward:', total_reward)
            total_reward = 0.

            if save_video:
                break
    pass

