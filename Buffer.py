import numpy as np
import torch


class Buffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        def transfer(data, first_dim=False):
            """
            transfer ndarray to torch.tensor,
            if `first_dim` is True, stack the ndarray so that the first dimension is `batch_size`,
            otherwise, the returned value is just a tensor with length of the original ndarray.
            """
            if first_dim:
                data = np.vstack(data)
            data = torch.from_numpy(data).float()
            return data

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = transfer(obs, first_dim=True)  # torch.Size([batch_size, state_dim])
        action = transfer(action, first_dim=True)  # torch.Size([batch_size, action_dim])
        reward = transfer(reward)  # just a tensor with length: batch_size
        next_obs = transfer(next_obs, first_dim=True)  # Size([batch_size, state_dim])
        done = transfer(done)  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size
