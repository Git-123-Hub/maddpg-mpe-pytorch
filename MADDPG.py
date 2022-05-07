import logging
import os
from copy import deepcopy

import numpy as np
import torch

from Agent import Agent
from Buffer import Buffer
import torch.nn.functional as F


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
    def __init__(self, obs_dim_list, act_dim_list, capacity, actor_lr, critic_lr, res_dir=None):
        # sum all the dims of each agent to get input dim for critic
        global_obs_dim = sum(obs_dim_list + act_dim_list)
        # create all the agents and corresponding replay buffer
        self.agents = []
        self.buffers = []
        for obs_dim, act_dim in zip(obs_dim_list, act_dim_list):
            self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr))
            self.buffers.append(Buffer(capacity, obs_dim, act_dim))
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
        # sample from all the replay buffer using the same index
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # and only the reward and done of the current agent is needed in the calculation
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

    def select_action(self, obs, *, explore=True):
        actions = []
        for n, agent in enumerate(self.agents):  # each agent select action according to their obs
            act = agent.action(obs[n], explore=explore).squeeze(0).detach().numpy()
            actions.append(act)
            self.logger.info(f'agent {n}, obs: {obs[n]} action: {act}')
        return actions

    def my_learn(self, batch_size, gamma):
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[0])
        # sample from all the replay buffer using the same index
        indices = np.random.choice(total_num, size=batch_size, replace=False)
        samples = []
        obs_list, act_list, next_obs_list, next_act_list = [], [], [], []
        for n, buffer in enumerate(self.buffers):
            transitions = buffer.sample(indices)
            samples.append(transitions)
            obs_list.append(transitions[0])
            act_list.append(transitions[1])
            next_obs_list.append(transitions[3])
            # calculate next_action using target_network and next_state
            next_act_list.append(self.agents[n].target_action(transitions[3]))

        # todo: maybe sample and learn in one loop
        # todo: 查看上面数据长度

        # update all agents
        for n, agent in enumerate(self.agents):
            # update critic
            states, actions, rewards, next_states, dones = samples[n]
            critic_value = agent.critic_value(obs_list, act_list)

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(next_obs_list, next_act_list)
            target_value = rewards + gamma * next_target_critic_value * (1 - dones)
            # target_value = rewards + gamma * next_target_critic_value  # todo: maybe remove dones

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action_list = []
            # # todo: check difference between act_list
            # for agent_name in self.agents.keys():  # loop over all the agents
            #     if agent_name == n:  # action of the current agent is calculated using its actor
            #         # todo: try with noise
            #         action = agent.action(states, explore=False)  # NOTE that NO noise
            #     else:  # action of other agents is from the samples
            #         action = samples[agent_name][1]
            #     action_list.append(action)
            action_list = deepcopy(act_list)  # todo: deepcopy????
            # action of the current agent is calculated using its actor
            action_list[n] = agent.action(states, explore=True)  # NOTE that NO noise
            actor_loss = -agent.critic_value(obs_list, action_list).mean()
            agent.update_actor(actor_loss)
            self.logger.info(f'agent{n}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def learn(self, batch_size, gamma):
        for i, agent in enumerate(self.agents):
            obs, act, reward_cur, next_obs, done_cur, next_act, obs_cur = self.sample(batch_size, i)
            # update critic
            critic_value = agent.critic_value(obs, act)

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(next_obs, next_act)
            target_value = reward_cur + gamma * next_target_critic_value * (1 - done_cur)
            # target_value = rewards + gamma * next_target_critic_value  # todo: maybe remove dones

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs_cur, model_out=True)
            act[i] = action
            actor_loss = -agent.critic_value(obs, act).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)  # todo: add loss_pse
            self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents:
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def load(self, file):
        pass
