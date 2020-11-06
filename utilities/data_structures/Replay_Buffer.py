from collections import namedtuple, deque
import random
import torch
import numpy as np
# import multiprocessing
from multiprocessing import Process
import time

class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer
        Args:
            states: ndarray
            actions: int
            rewards: flaot64
            next_states: ndarray
            dones: bool
        """
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)
   
    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences
            
    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Time_Seq_Replay_Buffer(Replay_Buffer):

    def __init__(self, buffer_size, batch_size, delta_time_target, target_num, delta_time_running, seed):
        Replay_Buffer.__init__(self, buffer_size, batch_size, seed)
        assert delta_time_target > delta_time_running
        self.delta_idx = int(delta_time_target / delta_time_running)
        self.target_num = target_num

        # self.process_pool = multiprocessing.Pool(3)

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size

        sample_idxs = np.random.randint(low=self.delta_idx * self.target_num, high=self.__len__()-1, size=batch_size)

        total_idxs = np.array([np.arange(start=sample_idx-(self.target_num - 1)*self.delta_idx, stop=sample_idx + 1,
                                step=self.delta_idx) for sample_idx in sample_idxs])
        # time0 = time.time()
        memory = []
        for idxs in total_idxs:
            current_time_seq_state = []
            next_time_seq_state = []
            for idx in idxs:
                experience = self.memory[idx]
                current_time_seq_state.append(experience.state)
                next_time_seq_state.append(experience.next_state)

            current_time_seq_state = np.stack(current_time_seq_state)
            next_time_seq_state = np.stack(next_time_seq_state)
            experience = self.experience(current_time_seq_state, experience.action, experience.reward, next_time_seq_state, experience.done)
            memory.append(experience)
        # print('time: %.6f', time.time() - time0)
        return memory

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences
    pass

if __name__ == '__main__':
    time_seq_buffer = Time_Seq_Replay_Buffer(2000, 32, 0.5, 5, 0.1, 0)
    for i in range(1000):
        state = np.ones(shape=(224, 224, 3), dtype=np.float32) * i
        action = i
        reward = i + 0.5
        next_state = state + 0.5
        dones = False

        time_seq_buffer.add_experience(state, action, reward, next_state, dones)

    for i in range(100):
        time0 = time.time()
        time_seq_memory = time_seq_buffer.sample(32)
        print(time.time() - time0)

    # buffer = np.random.uniform(0, 255, (1000, 224, 224, 3))
    # idxs = np.random.randint(0, 1000, (32, 5))
    # a = buffer[idxs]
    pass

