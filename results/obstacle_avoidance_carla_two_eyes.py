import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym

from agents.actor_critic_agents.A2C import A2C
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C
from agents.policy_gradient_agents.PPO import PPO
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets_2_EYE
from agents.policy_gradient_agents.REINFORCE import REINFORCE

## envs import ##
from environments.carla_enviroments import env_v1_ObstacleAvoidance
# from environments.carla_enviroments import env_v1_ObstacleAvoidance

config = Config()
config.seed = 1
config.environment = gym.make("ObstacleAvoidance-v1")
config.num_episodes_to_run = 4000
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
    "DQN_Agents": {
        "learning_rate": 1e-2,
        "batch_size": 8,
        "buffer_size": 4000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1.0,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "gradient_clipping_norm": None,
        "learning_iterations": 1,
        "clip_rewards": False,

        ## useless
    }
}

if __name__ == "__main__":
    # AGENTS = [SAC_Discrete, DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
    #           DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
    AGENTS = [DQN_With_Fixed_Q_Targets_2_EYE]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
    pass