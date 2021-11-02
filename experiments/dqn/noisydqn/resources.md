# Noisy Dqn Extension

## Demos
### Python Demos
* [demo1](https://github-dotcom.gateway.web.tr/RoyalSkye/Atari-DRL/blob/master/Noisy_DQN/agent_noisy.py)
* [demo2](https://github-dotcom.gateway.web.tr/deligentfool/dqn_zoo/blob/master/Noisy%20DQN/noisy_dqn.py)
* [demo3](https://github.com/qfettes/DeepRL-Tutorials/blob/master/05.DQN-NoisyNets.ipynb)
* [demo4](https://github.com/cyoon1729/deep-Q-networks/blob/master/noisyDQN/noisy_dqn.py)

### C++ Demos
* [demo1](https://github.com/navneet-nmk/Pytorch-RL-CPP/blob/master/noisy.h)

## Erros
### Call resetNoise in NoisyLinear->forward()
In DoubleDqn, when behavior network forward once for trajectory collection, forward once for max action of *next_state*.
If *resetNoise()* called in *forward()* method, then version of *epsilon* updated. 
In backward, the Q<sub>current</sub> was calculated by version0, while the current buffer of *epsilon* has been updated into version1.
Then backward failed.