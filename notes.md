# Notes
## module.eval()
module.eval() and module.train() may produce different output with same module and same input 
as some special modules behavior differently in different mode. 
The common modules affected are bm modules and dropout modules. 

** Why BM and Dropout works different? **

[Ref](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
## eval(), NoGradGuard, detach()?
## .requires_grad_()
.requires_grad_() does not be inherited automatically by computation.

Refer to _test/trials/testtensor.cpp/testXGrad()/y_
## Kernel
Every operation performed on Tensor s creates a new function object, that performs the computation, and records that it happened.
## Module 
*Module* class or struct, how it cooperates with *shared_ptr*
## NAN detection
torch.autograd.detect_anomaly
## Dropout
John Schulman has said dropout doesnt really help in rl.
Rl is very sensitive to learning rate. 0.01 is too high, 1e-7 is too low
## Maxium Overestimation
* Maxium bias always exists due to estimation error
* Uniformly distributed overestimation does not affect policy searching
* But uniform overestimation distribution can not be assumed
* The overestimation error would then cause local-optimal policy or converge
* And bootstrapping distributed the error everywhere
* So, target model and double DQN created
* While in a lots of episodic cases, the estimated values would be corrected finally by terminal state true value
* So sometimes DQN may learn slower than DDQN but still get a not bad result

### Ref
* Sutton ed2 ch6.7
* [Double Dqn](https://arxiv.org/pdf/1509.06461.pdf)

## Off-policy
Q-learning is off-policy as:
In this case, the learned action-value function, Q, directly approximates q ⇤ , the optimal
action-value function, independent of the policy being followed. This dramatically
simplifies the analysis of the algorithm and enabled early convergence proofs. The policy
still has an e↵ect in that it determines which state–action pairs are visited and updated.
However, all that is required for correct convergence is that all pairs continue to be
updated. As we observed in Chapter 5, this is a minimal requirement in the sense that
any method guaranteed to find optimal behavior in the general case must require it.

SARSA is on-policy algorithm, as it takes Q(S', A') where A' is chosen by current policy

* Sutton ed2 ch6.5
* [VS.](https://analyticsindiamag.com/reinforcement-learning-policy/#:~:text=An%20off%2Dpolicy%2C%20whereas%2C,is%20used%20to%20make%20decisions.)

## Bias Node
* [Why are bias nodes used in neural networks?](https://stats.stackexchange.com/questions/185911/why-are-bias-nodes-used-in-neural-networks)
* *Finally, the tuned version uses a single shared bias for all action values in the top layer of the network.* [Double Dqn](https://arxiv.org/pdf/1509.06461.pdf)
* [Bias Layer in Pytorch](https://discuss.pytorch.org/t/learnable-bias-layer/4221)

## How clip affects backpropogation?
If loss clipped, backpropagation failed

## Network parameter initialization cause worse result in a2c beginning
Kaiming initiation.