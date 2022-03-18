import logging

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
    def __init__(self, obs_dim_list, act_dim_list, capacity, batch_size, actor_lr, critic_lr):
        # sum all the dims of each agent to get input dim for critic
        global_obs_dim = sum(obs_dim_list + act_dim_list)
        # create all the agents and corresponding replay buffer
        self.agents = []
        self.buffers = []
        for obs_dim, act_dim in zip(obs_dim_list, act_dim_list):
            self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr))
            self.buffers.append(Buffer(capacity, obs_dim, act_dim))
        self.batch_size = batch_size
        self.logger = setup_logger('maddpg.log')

    def add(self, obs, actions, rewards, next_obs, dones):
        """add experience to buffer"""
        for n, buffer in enumerate(self.buffers):
            buffer.add(obs[n], actions[n], rewards[n], next_obs[n], dones[n])

    def select_action(self, obs, *, explore=True):
        actions = []
        for n, agent in enumerate(self.agents):  # each agent select action according to their obs
            act = agent.action(obs[n], explore=explore).squeeze(0).detach().numpy()
            actions.append(act)
            self.logger.info(f'agent {n}, action: {act}')
        return actions

    def learn(self):
        pass

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau """
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents:
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def load(self, file):
        pass
