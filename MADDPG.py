from Agent import Agent


class MADDPG:
    def __init__(self, obs_dim_list, act_dim_list, capacity, batch_size, actor_lr, critic_lr):
        # sum all the dims of each agent to get input dim for critic
        global_obs_dim = sum(obs_dim_list + act_dim_list)
        # create all the agents
        self.agents = []
        for obs_dim, act_dim in zip(obs_dim_list, act_dim_list):
            self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr))
        # create all the buffers
        self.buffers = []

    def add(self):
        """add experience to buffer"""
        pass

    def select_action(self):
        pass

    def learn(self):
        pass
