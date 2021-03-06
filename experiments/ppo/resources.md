# PPO Resources

## Reference

* [PPO Interpretation](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl)

* [CartPole PPO](https://github.com/4kasha/CartPole_PPO)

* [A pong params ref](https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/pong-ppo.yaml)

* [GAE](https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737)

* [Some math](https://avandekleut.github.io/a2c/)

* [A Continuous Pytorch PPO](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py)

* [RLlib ppo](https://docs.ray.io/en/master/rllib-algorithms.html#proximal-policy-optimization-ppo)

* [Ray Project](https://github.com/ray-project/rl-experiments)

* [Spinup](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

* [Hyperparameters range](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe)

* [KL](http://joschu.net/blog/kl-approx.html)

* [Value clip](https://github.com/openai/baselines/issues/91)

* [PPO some feedback](https://github.com/openai/baselines/issues/445)

* [PPO value loss](https://github.com/openai/baselines/issues/91)

* [PPO pong 1](http://www.sagargv.com/blog/pong-ppo/)

* [PPO stackoverflow](https://stackexchange.com/search?q=ppo)

* [A Zhihu reproduce](https://zhuanlan.zhihu.com/p/50322028)

* [hints from stable-baselines3 ppo code reading](https://blog.csdn.net/jinzhuojun/article/details/80417179)

* [About lambda](https://slm-lab.gitbook.io/slm-lab/using-slm-lab/search-spec-ppo-on-breakout)

* [Clip objective function](https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view)

* [A simple implementation codes reading](https://towardsdatascience.com/a-graphic-guide-to-implementing-ppo-for-atari-games-5740ccbe3fbc)

* [OpenAI default hyperparameters](https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py)

## TODO

* Distributed PPO

* How clip works in backward?

* Review GAE: how it balanced bias and variance?

* Retrace

## Advanced (Not to read yet)

* [Deceptive Gradient](https://arxiv.org/pdf/2006.08505.pdf)

* Quality Deversity

* [Exploration Topic](https://stackoverflow.com/questions/63047930/reinforcement-learning-driving-around-objects-with-ppo)

* ACKTR

## Notes
?? ?? ?? ??? ??? ???
### TD(??) Value Estimation
[Reinforcement Learning ??? TD(??) Introduction(1)](https://towardsdatascience.com/reinforcement-learning-td-%CE%BB-introduction-686a5e4f4e60)

[Actor-Critic Methods, Advantage Actor-Critic (A2C) and Generalized Advantage Estimation (GAE)](https://avandekleut.github.io/a2c/)

[Generalized Advantage Estimate: Maths and Code](https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737)

#### GAE Advantage side
* g<sub>t</sub>: GAE Advantage of time step t.
* V<sub>t</sub>: V[S<sub>t</sub>]
* k step involved

??<sub>t</sub> = R<sub>t</sub> + ?? * V<sub>t+1</sub> - V<sub>t</sub>

g<sub>t</sub> = ??<sub>t</sub> + ?? * ?? * g<sub>t+1</sub>

= R<sub>t</sub> + ?? * V<sub>t+1</sub> - V<sub>t</sub> + ?? * ?? * g<sub>t+1</sub>

= R<sub>t</sub> + ?? * V<sub>t+1</sub> - V<sub>t</sub> + ?? * ?? * (R<sub>t+1</sub> + ?? * V<sub>t+2</sub> - V<sub>t+1</sub> + ?? * ?? * g<sub>t+2</sub>)

= R<sub>t</sub> + ?? * ?? * R<sub>t+1</sub> + ??<sup>2</sup>??V<sub>t+2</sub> + ??(1 - ??)V<sub>t+1</sub> - V<sub>t</sub> + (????)<sup>2</sup>g<sub>t+3</sub>

= ...

= ???<sub>i=0</sub><sup>k</sup> ??<sup>i</sup>??<sup>i</sup>R<sub>t+i</sub> + ???<sub>i=1</sub><sup>k+1</sup>??<sup>i</sup>??<sup>i-1</sup>(1-??)V<sub>t+i</sub> - V<sub>t</sub>

By SB3 implementation:

V_target<sub>t</sub> = g<sub>t</sub> + V_estimation<sub>t</sub>

= ???<sub>i=0</sub><sup>k</sup> ??<sup>i</sup>??<sup>i</sup>R<sub>t+i</sub> + ???<sub>i=1</sub><sup>k+1</sup>??<sup>i</sup>??<sup>i-1</sup>(1-??)V<sub>t+i</sub>

= B(R) + C(V)

#### TD(??) side
* G<sub>t+i</sub>: Return of TD(i) of time step t 

G<sub>t</sub> = R<sub>t</sub> + ??V<sub>t+1</sub>

G<sub>t+1</sub> = R<sub>t</sub> + ??R<sub>t+1</sub> + ??<sup>2</sup>V<sub>t+1</sub>

G<sub>t+k</sub> + ???<sup>i</sup><sub>0~k</sub>??<sup>i</sup>R<sub>t+i</sub> + ??<sup>k+1</sup>V<sub>t+k+1</sub>

G(??)<sub>t</sub> = (1 - ??)???<sup>i</sup><sub>0~k</sub>G<sub>t+i</sub>

= (1 - ??) * F

F = ???<sup>i</sup><sub>0~k</sub>G<sub>t+i</sub>

= (R<sub>t</sub> + ??V<sub>t+1</sub>)
    + ??(R<sub>t</sub> + ??R<sub>t+1</sub> + ??<sup>2</sup>V<sub>t+1</sub>)
    + ??<sup>2</sup>(R<sub>t</sub> + ??R<sub>t+1</sub> + ??<sup>2</sup>R<sub>t+2</sub> + ??<sup>2</sup>V<sub>t+1</sub>)
    + ...
    + ??<sup>k</sup>(R<sub>t</sub> + ??R<sub>t+1</sub> + ... + ??<sup>k</sup>R<sub>t+k</sub> + ??<sup>2</sup>V<sub>t+1</sub>)

H(R) = (1 + ?? + ??<sup>2</sup> + ... + ??<sup>k</sup>)R<sub>t</sub>
    + ????(1 + ?? + ??<sup>2</sup> + ... + ??<sup>k-1</sup>)R<sub>t+1</sub>
    + ...
    + ??<sup>k</sup>??<sup>k</sup>R<sub>t+k</sub>

Y(V) = ???<sup>i</sup><sub>1~k+1</sub>??<sup>i</sup>??<sup>i-1</sup>V<sub>t+i</sub>

F = H(R) + Y(V)

G(??) = (1 - ??)F = (1 - ??)H(R) + (1 - ??)Y(V)

#### Sum up
(1 - ??)Y(V) = C(V)

Suppose k->???, make approximation:

(1 + ?? + ??<sup>2</sup> + ... + ??<sup>k-i</sup>) = 1 / (1 - ??)   ???i

==> H(R) = (???<sup>i</sup><sub>0~k</sub>??<sup>i</sup>??<sup>i</sup>R<sub>t+i</sub>) / (1 - ??)

==> (1 - ??)H(R) = ???<sup>i</sup><sub>0~k</sub>??<sup>i</sup>??<sup>i</sup>R<sub>t+i</sub>

==> (1 - ??)H(R) = B(R)

So, G<sub>t</sub> of TD(??) is the same as V_target<sub>t</sub> of GAE, that is the how SB3 implemented.



