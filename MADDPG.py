import logging

from Agent import Agent
from buffer import buffer


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
            self.buffers.append(buffer(capacity, obs_dim, act_dim))
        self.batch_size = batch_size
        self.logger = setup_logger('maddpg.log')

    def add(self):
        """add experience to buffer"""
        pass

    def select_action(self):
        pass

    def learn(self):
        pass

    def update_target(self, tau):
        pass

    def load(self, file):
        pass
