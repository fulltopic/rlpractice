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
λ γ δ ∑ ∞ ∀
### TD(λ) Value Estimation
[Reinforcement Learning — TD(λ) Introduction(1)](https://towardsdatascience.com/reinforcement-learning-td-%CE%BB-introduction-686a5e4f4e60)

[Actor-Critic Methods, Advantage Actor-Critic (A2C) and Generalized Advantage Estimation (GAE)](https://avandekleut.github.io/a2c/)

[Generalized Advantage Estimate: Maths and Code](https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737)

#### GAE Advantage side
* g<sub>t</sub>: GAE Advantage of time step t.
* V<sub>t</sub>: V[S<sub>t</sub>]
* k step involved

δ<sub>t</sub> = R<sub>t</sub> + γ * V<sub>t+1</sub> - V<sub>t</sub>

g<sub>t</sub> = δ<sub>t</sub> + γ * λ * g<sub>t+1</sub>

= R<sub>t</sub> + γ * V<sub>t+1</sub> - V<sub>t</sub> + γ * λ * g<sub>t+1</sub>

= R<sub>t</sub> + γ * V<sub>t+1</sub> - V<sub>t</sub> + γ * λ * (R<sub>t+1</sub> + γ * V<sub>t+2</sub> - V<sub>t+1</sub> + γ * λ * g<sub>t+2</sub>)

= R<sub>t</sub> + γ * λ * R<sub>t+1</sub> + γ<sup>2</sup>λV<sub>t+2</sub> + γ(1 - λ)V<sub>t+1</sub> - V<sub>t</sub> + (γλ)<sup>2</sup>g<sub>t+3</sub>

= ...

= ∑<sub>i=0</sub><sup>k</sup> γ<sup>i</sup>λ<sup>i</sup>R<sub>t+i</sub> + ∑<sub>i=1</sub><sup>k+1</sup>γ<sup>i</sup>λ<sup>i-1</sup>(1-λ)V<sub>t+i</sub> - V<sub>t</sub>

By SB3 implementation:

V_target<sub>t</sub> = g<sub>t</sub> + V_estimation<sub>t</sub>

= ∑<sub>i=0</sub><sup>k</sup> γ<sup>i</sup>λ<sup>i</sup>R<sub>t+i</sub> + ∑<sub>i=1</sub><sup>k+1</sup>γ<sup>i</sup>λ<sup>i-1</sup>(1-λ)V<sub>t+i</sub>

= B(R) + C(V)

#### TD(λ) side
* G<sub>t+i</sub>: Return of TD(i) of time step t 

G<sub>t</sub> = R<sub>t</sub> + γV<sub>t+1</sub>

G<sub>t+1</sub> = R<sub>t</sub> + γR<sub>t+1</sub> + γ<sup>2</sup>V<sub>t+1</sub>

G<sub>t+k</sub> + ∑<sup>i</sup><sub>0~k</sub>γ<sup>i</sup>R<sub>t+i</sub> + γ<sup>k+1</sup>V<sub>t+k+1</sub>

G(λ)<sub>t</sub> = (1 - λ)∑<sup>i</sup><sub>0~k</sub>G<sub>t+i</sub>

= (1 - λ) * F

F = ∑<sup>i</sup><sub>0~k</sub>G<sub>t+i</sub>

= (R<sub>t</sub> + γV<sub>t+1</sub>)
    + λ(R<sub>t</sub> + γR<sub>t+1</sub> + γ<sup>2</sup>V<sub>t+1</sub>)
    + λ<sup>2</sup>(R<sub>t</sub> + γR<sub>t+1</sub> + γ<sup>2</sup>R<sub>t+2</sub> + γ<sup>2</sup>V<sub>t+1</sub>)
    + ...
    + λ<sup>k</sup>(R<sub>t</sub> + γR<sub>t+1</sub> + ... + γ<sup>k</sup>R<sub>t+k</sub> + γ<sup>2</sup>V<sub>t+1</sub>)

H(R) = (1 + λ + λ<sup>2</sup> + ... + λ<sup>k</sup>)R<sub>t</sub>
    + γλ(1 + λ + λ<sup>2</sup> + ... + λ<sup>k-1</sup>)R<sub>t+1</sub>
    + ...
    + λ<sup>k</sup>γ<sup>k</sup>R<sub>t+k</sub>

Y(V) = ∑<sup>i</sup><sub>1~k+1</sub>γ<sup>i</sup>λ<sup>i-1</sup>V<sub>t+i</sub>

F = H(R) + Y(V)

G(λ) = (1 - λ)F = (1 - λ)H(R) + (1 - λ)Y(V)

#### Sum up
(1 - λ)Y(V) = C(V)

Suppose k->∞, make approximation:

(1 + λ + λ<sup>2</sup> + ... + λ<sup>k-i</sup>) = 1 / (1 - λ)   ∀i

==> H(R) = (∑<sup>i</sup><sub>0~k</sub>γ<sup>i</sup>λ<sup>i</sup>R<sub>t+i</sub>) / (1 - λ)

==> (1 - λ)H(R) = ∑<sup>i</sup><sub>0~k</sub>γ<sup>i</sup>λ<sup>i</sup>R<sub>t+i</sub>

==> (1 - λ)H(R) = B(R)

So, G<sub>t</sub> of TD(λ) is the same as V_target<sub>t</sub> of GAE, that is the how SB3 implemented.



