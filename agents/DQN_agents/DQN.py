from collections import Counter
import os
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer
import glob
import re

class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.SGD(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], weight_decay=5e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            # print('state:', self.state)
            # self.environment.render()
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(self.hyperparameters["learning_iterations"]):
                    try:
                        self.environment.pause()
                        # print('pause')
                        self.learn()
                        self.environment.resume()
                        # print('resume')
                    except:
                        self.learn()
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train() #puts network back in training mode


        force_explore = self.config.force_explore_mode and self.need_to_force_explore()

        if force_explore:
            print('explore...')

        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number,
                                                                                    "force_explore":force_explore})
        # self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        # loss = F.mse_loss(Q_expected, Q_targets)

        loss = nn.MSELoss(size_average=False)(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

    def locally_save_policy(self, best=True, episode=None):
        if self.agent_name != "DQN":
            state = {'episode': self.episode_number,
                     'q_network_local': self.q_network_local.state_dict(),
                     'q_network_target': self.q_network_target.state_dict()}
        else:
            state = {'episode': self.episode_number,
                     'q_network_local': self.q_network_local.state_dict()}

        model_root = os.path.join('Models', self.config.env_title, self.agent_name, self.config.log_base)
        if not os.path.exists(model_root):
            os.makedirs(model_root)

        if best:
            last_best_file = glob.glob(os.path.join(model_root, 'rolling_score*'))
            if last_best_file:
                os.remove(last_best_file[0])

            save_name = model_root + "/rolling_score_%.4f.model"%(self.rolling_results[-1])
            torch.save(state, save_name)
            self.logger.info('Model-%s save success...' % (save_name))
        else:
            save_name = model_root + "/%s_%d.model" % (self.agent_name, self.episode_number)
            torch.save(state, save_name)
            self.logger.info('Model-%s save success...' % (save_name))

    def load_resume(self, resume_path):
        save = torch.load(resume_path)
        if self.agent_name != "DQN":
            q_network_local_dict = save['q_network_local']
            q_network_target_dict = save['q_network_target']
            self.q_network_local.load_state_dict(q_network_local_dict, strict=True)
            self.q_network_target.load_state_dict(q_network_target_dict, strict=True)
        else:
            q_network_local_dict = save['q_network_local']
            self.q_network_local.load_state_dict(q_network_local_dict, strict=True)
        self.logger.info('load resume model success...')

        file_name = os.path.basename(resume_path)
        episode_str = re.findall(r"\d+\.?\d*", file_name)[0]
        episode_list = episode_str.split('.')
        if not episode_list[1]:
            episode = episode_list[0]
        else:
            episode = 0

        if not self.config.retrain:
            self.episode_number = episode
        else:
            self.episode_number = 0