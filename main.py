import argparse

from multiagent import scenarios
from multiagent.environment import MultiAgentEnv

from MADDPG import MADDPG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='name of the environment',
                        choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
                                 'simple_speaker_listener', 'simple_spread', 'simple_tag',
                                 'simple_world_comm'])
    parser.add_argument('--episode-length', type=int, default=100, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=5000, help='total number of episode')
    parser.add_argument('--buffer-capacity', default=int(1e6))
    parser.add_argument('--batch-size', default=1000)
    parser.add_argument('--actor-lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic-lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    # create env
    scenario = scenarios.load(f'{args.env}.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    obs = env.reset()
    # get dimension info about observation and action
    obs_dim_list = []
    for obs_space in env.observation_space:  # continuous observation
        obs_dim_list.append(obs_space.shape[0])  # Box
    act_dim_list = []
    for act_space in env.action_space:  # discrete action
        act_dim_list.append(act_space.n)  # Discrete

    maddpg = MADDPG(obs_dim_list, act_dim_list, args.buffer_capacity, args.batch_size,
                    args.actor_lr, args.critic_lr)
