import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    def __init__(self, obs_dim_list, act_dim_list, capacity, actor_lr, critic_lr, res_dir=None, device=None):
        """
        :param obs_dim_list: list of observation dimension of each agent
        :param act_dim_list: list of action dimension of each agent
        :param capacity: capacity of the replay buffer
        :param res_dir: directory where log file and all the data and figures will be saved
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f'training on device: {self.device}')
        # sum all the dims of each agent to get input dim for critic
        global_obs_dim = sum(obs_dim_list + act_dim_list)
        # create all the agents and corresponding replay buffer
        self.agents = []
        self.buffers = []
        for obs_dim, act_dim in zip(obs_dim_list, act_dim_list):
            self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, self.device))
            self.buffers.append(Buffer(capacity, obs_dim, act_dim, self.device))
        if res_dir is None:
            self.logger = setup_logger('maddpg.log')
        else:
            self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, obs, actions, rewards, next_obs, dones):
        """add experience to buffer"""
        for n, buffer in enumerate(self.buffers):
            buffer.add(obs[n], actions[n], rewards[n], next_obs[n], dones[n])

    def sample(self, batch_size, agent_index):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[0])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs_list, act_list, next_obs_list, next_act_list = [], [], [], []
        reward_cur, done_cur, obs_cur = None, None, None
        for n, buffer in enumerate(self.buffers):
            obs, action, reward, next_obs, done = buffer.sample(indices)
            obs_list.append(obs)
            act_list.append(action)
            next_obs_list.append(next_obs)
            # calculate next_action using target_network and next_state
            next_act_list.append(self.agents[n].target_action(next_obs))
            if n == agent_index:  # reward and done of the current agent
                obs_cur = obs
                reward_cur = reward
                done_cur = done

        return obs_list, act_list, reward_cur, next_obs_list, done_cur, next_act_list, obs_cur

    def select_action(self, obs):
        actions = []
        for n, agent in enumerate(self.agents):  # each agent select action according to their obs
            o = torch.from_numpy(obs[n]).unsqueeze(0).float().to(self.device)  # torch.Size([1, state_size])
            # Note that the result is tensor, convert it to ndarray before input to the environment
            act = agent.action(o).squeeze(0).detach().cpu().numpy()
            actions.append(act)
            # self.logger.info(f'agent {n}, obs: {obs[n]} action: {act}')
        return actions

    def learn(self, batch_size, gamma):
        for i, agent in enumerate(self.agents):
            obs, act, reward_cur, next_obs, done_cur, next_act, obs_cur = self.sample(batch_size, i)
            # update critic
            critic_value = agent.critic_value(obs, act)

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(next_obs, next_act)
            target_value = reward_cur + gamma * next_target_critic_value * (1 - done_cur)

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs_cur, model_out=True)
            act[i] = action
            actor_loss = -agent.critic_value(obs, act).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents:
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
