import torch
import torch.nn.functional as F
from agents.DQN_agents.DDQN import DDQN
from utilities.data_structures.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer
import os
import re

class DDQN_With_Prioritised_Experience_Replay(DDQN):
    """A DQN agent with prioritised experience replay"""
    agent_name = "DDQN with Prioritised Replay"

    def __init__(self, config):
        DDQN.__init__(self, config)
        self.memory = Prioritised_Replay_Buffer(self.hyperparameters, config.seed)

        if config.resume:
            self.load_resume(config.resume_path)

    def learn(self):
        """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
        sampled_experiences, importance_sampling_weights = self.memory.sample()
        states, actions, rewards, next_states, dones = sampled_experiences
        loss, td_errors = self.compute_loss_and_td_errors(states, next_states, rewards, actions, dones, importance_sampling_weights)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyperparameters["tau"])
        self.memory.update_td_errors(td_errors.squeeze(1))

    def save_experience(self):
        """Saves the latest experience including the td_error"""
        max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
        self.memory.add_experience(max_td_error_in_experiences, self.state, self.action, self.reward, self.next_state, self.done)

    def compute_loss_and_td_errors(self, states, next_states, rewards, actions, dones, importance_sampling_weights):
        """Calculates the loss for the local Q network. It weighs each observations loss according to the importance
        sampling weights which come from the prioritised replay buffer"""
        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        loss = loss * importance_sampling_weights
        loss = torch.mean(loss)
        td_errors = Q_targets.data.cpu().numpy() - Q_expected.data.cpu().numpy()
        return loss, td_errors

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

        file_name = os.path.basename(resume_path)
        episode_str = re.findall(r"\d+\.?\d*", file_name)[0]
        episode_list = episode_str.split('.')
        if not episode_list[1]:
            episode = episode_list[0]
        else:
            episode = 0

        if not self.config.retrain:
            self.episode_number = int(episode)
        else:
            self.episode_number = 0

        self.logger.info('load resume model-%s success, the training episode will be started from %d...'%(resume_path, self.episode_number))