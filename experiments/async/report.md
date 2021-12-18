# Asynchronized Actor-Critic

## Ref
* [an A3C demo](https://github.com/ikostrikov/pytorch-a3c)

## Test Cases
### Asynchronized by grad parameters
It applied original A3C algorithm.
* Multiple workers and one updater. Each worker or updater is executed in different process/ 
* Each worker and updater holds a network instance
* Workers send grad parameters to updater, updater sends network parameters to worker.
* Workers and updater communicate by network(TCP)

#### testcart0
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|0.001|
|valueCoef|0.5|
|entropyCoef|0.01|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|32|
|maxStep|8|
|gradSyncStep|32|
|targetUpdateStep|32|

TODO: To test again

#### testpong1
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|0.001|
|valueCoef|0.5|
|entropyCoef|0.01|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|32|
|maxStep|8|
|gradSyncStep|16|
|targetUpdateStep|16|
|updateThread|2|

The worker calculates the grad parameters and accumulates (sum up) these grad parameters 
produced in *gradSyncStep*.
The server(updater) receives all grad parameters from all the workers and simply update the network by *optimizer.step()*.
The server is serviced by 2 threads without synchronization. 

This test failed:

![a3cpong1_loss](./images/a3cpong1_loss.jpg)

![a3cpong1_test](./images/a3cpong1_test.jpg)

#### testpong2
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|1e-4|
|valueCoef|0.5|
|entropyCoef|0.01|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|gradSyncStep|10|
|targetUpdateStep|10|
|updateThread|2|

It worked.

![a3cpong2_loss](./images/a3cpong2_loss.jpg)

![a3cpong2_train](./images/a3cpong2_train.jpg)

![a3cpong2_test](./images/a3cpong2_test.jpg)

#### testpong3
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|1e-4|
|valueCoef|0.5|
|entropyCoef|0.01, 0.01, 0.01, 0.005|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|gradSyncStep|20|
|targetUpdateStep|40|
|updateThread|1|

With more workers, variable *entropyCoef*, only one server thread, synchronization interval increased.

![a3cpong3_loss](./images/a3cpong3_loss.jpg)

![a3cpong3_train](./images/a3cpong3_train.jpg)

![a3cpong3_test](./images/a3cpong3_test.jpg)

This case is better than __testpong2__: the training reward reached maximum after about 300k steps,
while in this case  the training reward got 21 after about 150k steps. Don't know which factor weighted.

#### testpong4
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|1e-4|
|valueCoef|0.5|
|entropyCoef|0.01, 0.02, 0.005|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|gradSyncStep|10|
|targetUpdateStep|20|
|updateThread|1|

The different from testpong3:
* 3 workers
* different *entropyCoef*
* shorter synchronization span  
* Updater received and update each grad parameter package and update all the received grads sequentially.

![a3cpong4_loss](./images/a3cpong4_loss.jpg)

![a3cpong4_train](./images/a3cpong4_train.jpg)

![a3cpong4_test](./images/a3cpong4_test.jpg)

With less worker than __testpong3__, this case reached 21 by about 100k steps. It might be caused by:
1. shorter synchronization span
2. synchronization of update to avoid cancel

#### testpong5
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|1e-4|
|valueCoef|0.5|
|entropyCoef|0.05, 0.01, 0.02, 0.005|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|gradSyncStep|10|
|targetUpdateStep|20|
|updateThread|2|

Core dump. Maybe caused by concurrent update.

So the potential cause #2 in previous case is less possible

#### testpong6
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|5e-4|
|valueCoef|0.5|
|entropyCoef|0.05, 0.01, 0.02, 0.005|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|gradSyncStep|5|
|targetUpdateStep|20|
|updateThread|1|
|#worker|3|

![a3cpong6_loss](./images/a3cpong6_loss.jpg)

![a3cpong6_train](./images/a3cpong6_train.jpg)

![a3cpong6_test](./images/a3cpong6_test.jpg)

It failed.

#### testpong7
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|1e-3|
|valueCoef|0.5|
|entropyCoef|0.05, 0.01, 0.02, 0.005|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|gradSyncStep|20|
|targetUpdateStep|40|
|updateThread|1|
|#worker|4|

![testpong7_test](./images/a3cpong7_test.jpg)

It failed.

### Asynchronized by network(model)
It applied original A3C algorithm.
* Multiple workers and no updater
* All workers share a same network
* Each worker runs forward phrase independently
* Each worker runs backward phrase by its own, but the (backward + update) were executed sequentially    
#### testpong9
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|1e-3|
|valueCoef|0.5|
|entropyCoef|0.01, 0.01, 0.02, 0.005|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|updateThread|4|
|#worker|4|

Difference from previous cases:
* Network shared in the same process, no networking, no replica, no extra synchronization required
* Each worker executes backward and update by itself, but the updates were synchronized in sequential way. 

![a3cpong9_loss](./images/a3cpong9_loss.jpg)

![a3cpong9_train](./images/a3cpong9_train.jpg)

![a3cpong9_test](./images/a3cpong7_train.jpg)

Compared to that of __testpong3__, performance of this case is much better. 
The number of steps taken to reach maximum is about half.
Compared to that of __testpong4__, the number of steps to reach maximum is about 3/4.

It might be because that the immediate update reduce the impact of off-policy trace to minimum.

#### testbr10
|name|value|
|----|-----|
|Optimizer|Adam|
|lr|1e-3|
|valueCoef|0.5|
|entropyCoef|0.01, 0.01, 0.02, 0.02, 0.005|
|maxGradNormClip|0.1|
|gamma|0.99|
|reward|[-1, 1]|
|batchSize|50|
|maxStep|5|
|updateThread|5|
|#worker|5|

It was a breakout test case.

The case followed the same architecture of __testpong9__. 

![a3cbr10_loss](./images/a3cbr10_loss.jpg)

![a3cbr10_train](./images/a3cbr10_train.jpg)

![a3cbr10_test](./images/a3cbr10_test.jpg)

* The training encountered something like catastrophic forgetting, but it recovered
* The *test/len* is in a shape of dramatic increase. It is because that the agent hang around last few bricks in each episode.
* The *train/epReward* reached 400 at about 6k episodes. After that, the test case run for another 10k extra episode, but it still failed to overcome the hang.
* The *epReward* was the summary of rewards of 5 lives. When the agent hung, an episode may end in less than 5 lives.
* Anyway, the test case got reward of 400 in about 120k training steps. Not bad performance.

### Asynchronized by trajectories

## Conclusion
* lr
* cost
* TODO: shared network