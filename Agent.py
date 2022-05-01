from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import gumbel_softmax, one_hot
from torch.optim import Adam
import torch.nn.functional as F


def my_gumbel_softmax(logits, tau=1, eps=1e-20):
    epsilon = torch.rand_like(logits)
    logits += -torch.log(-torch.log(epsilon + eps) + eps)
    return F.softmax(logits / tau, dim=-1)


class Agent:
    """single agent in MADDPG"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        # the actor output logit of each action
        self.actor = MLPNetwork(obs_dim, act_dim)
        # critic input all the states and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def action(self, obs, *, explore):
        # this method is called in the following two cases:
        # a) interact with the environment, where input is a numpy.ndarray
        # NOTE that the output is a tensor, you have to convert it to ndarray before input to the environment
        # b) when update actor, calculate action using actor and states,
        # which is sampled from replay buffer with size: torch.Size([batch_size, state_dim])

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).unsqueeze(0).float()  # torch.Size([1, state_size])
        action = self.actor(obs)  # torch.Size([batch_size, action_size])
        explore = True  # todo：maybe remove explore

        if explore:
            # action = gumbel_softmax(action, tau=1, hard=False)
            action = my_gumbel_softmax(action)
            # if hard=True, the returned samples will be discretized as one-hot vectors
        else:
            # choose action with the biggest actor_output(logit)
            max_index = action.max(dim=1)[1]
            action = one_hot(max_index, num_classes=action.size(1))
        return action  # onehot tensor with size: torch.Size([batch_size, action_size])

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        action = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = gumbel_softmax(action, hard=False)
        action = my_gumbel_softmax(action)
        return action.squeeze(0).detach()  # onehot tensor with size: torch.Size([batch_size, action_size])
        # NOTE that I didn't use noise during this procedure
        # so I just choose action with the biggest actor_output(logit)
        max_index = action.max(dim=1)[1]
        action = one_hot(max_index, num_classes=action.size(1)).detach()
        return action  # onehot tensor with size: torch.Size([batch_size, action_size])

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU(), last_layer=None):
        super(MLPNetwork, self).__init__()

        modules = [
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ]
        if last_layer is not None:
            modules.append(last_layer)
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
