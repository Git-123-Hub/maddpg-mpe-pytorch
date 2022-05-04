# MADDPG using PyTorch and MPE

- [paper](https://arxiv.org/pdf/1706.02275.pdf)
- [MPE](https://github.com/openai/multiagent-particle-envs)

# Trouble shooting

`ImportError: cannot import name 'prng' from 'gym.spaces'`

after install multiagent-particle-envs, install an older version of gym

```shell
pip uninstall gym
pip install gym==0.10.5
```
