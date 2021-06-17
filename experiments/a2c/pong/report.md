# A2C PongNoFrameskip-v4

__TODO__
* Network description
* X label is number of update

## AirACCnnNet
### test0
Parameters copied from StableBaselines3

|name|value|
|----|-----|
|env_num|16|
|#step|5|
|batchSize|16 * 5|
|entropyCoef|0.01|
|valueCoef|0.25|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It learned very slowly.

![test0_reward](./images/test0_reward.jpg)

The entropy is rather unstable. It seemed that whenever agent thought it learned something,
it was over-confident. And the estimated optimal action was proved to be a very bad choice immediately.
Then agent returned to random guess. So the entropy increased and decreased dramatically. 
While overall, it intended to decrease. That is, the agent had learned something, but in a very hard way.

![test0_entropy](./images/test0_entropy.jpg)

The action_loss/total_loss ratio seemed proved it. 
The first black arrow shows that there was almost no action_loss. 
The second black arrow shows that at that time, action_loss = 0 and total_loss = 0.

![test0_ratio](./images/test0_action_total.jpg)

### test1
Continue to see if the unstable learning was just because of unlucky start.

|name|value|
|----|-----|
|env_num|16|
|#step|5|
|batchSize|16 * 5|
|entropyCoef|0.01|
|valueCoef|0.25|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

While it just failed to learn.

![test1_reward](./images/test1_reward.jpg)

![test1_entropy](./images/test1_entropy.jpg)

### test2
Increase _valueCoef_ to push estimation of V(s), hope to help decrease variance (according to reward shaping)

|name|value|
|----|-----|
|env_num|16|
|#step|5|
|batchSize|16 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Seemed improved, while still bad:

![test2_reward](./images/test2_reward.jpg)

![test2_test1_entropy](./images/test2_test1_entropy.jpg)

### test3/test4
Continue test2 parameters for more updates:

|name|value|
|----|-----|
|env_num|16|
|#step|5|
|batchSize|16 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Still failed:

![test3_reward](./images/test3_reward.jpg)
![test4_reward](./images/test4_reward.jpg)

![test4_entropy](./images/test4_entropy.jpg)

### test5
Decreased *entropyCoef* hope to reduce exploration and variance

|name|value|
|----|-----|
|env_num|16|
|#step|5|
|batchSize|16 * 5|
|entropyCoef|0.005|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Seemed a catastrophical forgetting:

![test5_reward_entropy](./images/test5_reward_entropy.png)

### test7
Increase _entropyCoef_ 

|name|value|
|----|-----|
|env_num|16|
|#step|5|
|batchSize|16 * 5|
|entropyCoef|0.02|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Failed to learn:

![test7_reward](./images/test7_reward.jpg)

### test6
Increased _env_num_ to reduce variance based on test3

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Compared to 16 env_num case (test4), 32 instances has better reward, although with similar trend.

![test6_test4_reward](./images/test6_test4_reward.png)

The entropy was still unstable, but there was no case that entropy remained 0 in a relatively long term (between 15,000 and 20,000 in blue in following figure). 

![test6_test4_entropy](./images/test6_test4_entropy.png)

### test8
Continue more updates based on test6

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Get positive average reward first time, but still dramatical decrease when the entropy diminished:

![test8_reward_entropy](./images/test8_reward_entropy.png)

### test9
Continue more updates based on test8

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Had less diminished entropy and bigger average reward. But there were still sudden catastrophe at the end of training. 

![test9_reward_entropy](./images/test9_reward_entropy.png)

### testtest9
A validation on test9 model.

It shows that agent had learned the policy on how to win the game with max reward,
but maybe not good at dealing with certain competitors:

![testtset9_reward](./images/testtest9_reward.jpg)

### test10
Reduced *entropyCoef* to check if negative reward cases in testtest9 was caused by too much exploration. 

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.005|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

The agent just did not converge:

![test10_reward](./images/test10_reward_ave.jpg)

![test10_entropy](./images/test10_entropy.jpg)

### test11/test14
Try an entropy between 0.01 and 0.005

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.008|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It was promising in short term:

![test11_entropy](./images/test11_entropy.jpg)

![test11_reward](./images/test11_reward.jpg)

But crashed after more updates:

![test14_entropy](./images/test14_entropy.jpg)

![test14_reward](./images/test14_reward.jpg)

### test12 
Recover entropy as 0.01 and try again:

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Still similar to test9

![test12_entropy](./images/test12_entropy.jpg)

![test12_reward](./images/test12_reward.jpg)

### test13
Descreased maxGradNormClip to make the agent update more carefully

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It did not help:

![test13_entropy](./images/test13_entropy.jpg)

![test13_reward](./images/test13_reward.jpg)

### test15
Extended maxstep before update

|name|value|
|----|-----|
|env_num|32|
|#step|8|
|batchSize|32 * 8|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It was worse:

![test15_entropy](./images/test15_entropy.jpg)

![test15_reward](./images/test15_reward.jpg)

## AirACHONet
Guessed that Stablebaselines3 has fine-tuned their codes 
so that the simple _AirACCnnNet_ could reach the optimal policy. 

Turned to _Deep Reinforcement Learning Hands-On_ to repeat its A2C case 
and hoped that the network with more cells could solve the problem.

### test17

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.1|
|optimizer|Adam|
|learningRate|0.001|
|eps|1e-3|

It worked:

![test17_reward](./images/test17_reward.jpg)

![test17_entropy](./images/test17_entropy.jpg)

### test18
Try Stablebaselines3 parameters:

|name|value|
|----|-----|
|env_num|32|
|#step|5|
|batchSize|32 * 5|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It indicated that the problem did not lay in network structure:

![test18_reward](./images/test18_reward.jpg)

![test18_entropy](./images/test18_entropy.jpg)

### test19
Extended maxStep:

|name|value|
|----|-----|
|env_num|32|
|#step|8|
|batchSize|32 * 8|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It was unstable, but learned:

![test19_entropy](./images/test19_entropy.jpg)

![test19_reward](./images/test19_reward.jpg)

### test20
Tried to increase maxStep and env_num to increase batchSize to improve stability. 
And to reduce valueCoef to push significance of action_loss

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.01|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It worked:

![test20_entropy](./images/test20_entropy.jpg)

![test20_reward](./images/test20_reward.jpg)

And actionLoss dominated overall loss:

![test20_ratio](./images/test20_action_loss_ratio.jpg)

### testtest20
Test model generated by test20. The agent had learned how to win.
In some games it won by max reward, some games it won by slim edge.

![testtest20](./images/testtest20_reward.jpg)

### test21
Increase valueCoef:

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.01|
|valueCoef|0.5|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

valueCoef is not a very critical factor. test21 still worked. But at update#10240, it was a bit slower than test20

![test21_reward](./images/test21_reward.jpg)

### t22
The slope of reward increasing is gentle after ave_reward = 10 reached.
Reduced entropyCoef to reduce exploration. Reduce valueCoef to push action_loss optimization.
Based on model generated by test20.

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.005|
|valueCoef|0.5|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Seemed that the agent trapped in sub-optimal policy:

![test22_reward](./images/test22_reward.jpg)

### test23
Recover parameters of test21 and continue with more updates:

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.01|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Although entropy decreased, reward performance did not improve:

![test23_reward](./images/test23_reward.jpg)

![test23_test22](./images/test22_test23_entropy.png)

### test24
Try smaller entropyCoef as model generated by test22 was supposed to be relatively confident.

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.01|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Entropy indicated that the model did exploared and adjusted

![test24_entroy](./images/test24_entropy.jpg)

But no improvement on reward:

![test24_reward](./images/test24_reward.jpg)

Seemed the model had reached its best.

### testtest24
And test result was not bad:

![testtest24](./images/testtest24_reward.jpg)

### test25
Try bigger valueCoef as accurate value estimation helps reducing bias (by reward shaping).
The model is based on model generated by test24

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.001|
|valueCoef|0.5|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

But rewward result was no better or worse than test24:

![test25_reward](./images/test25_reward.jpg)

### test26
Try even smaller entropyCoef:
The model is based on model generated by test24

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.0005|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

It did not work:

![test26](./images/test26_reward.jpg)

### test27
Decrease maxStep and entropyCoef.

|name|value|
|----|-----|
|env_num|50|
|#step|5|
|batchSize|50 * 5|
|entropyCoef|0.001|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

The reward was decreasing. Don't know which factor weighted. Or maybe it was adjusting to new parameters, it might be able to improve with more updates.

![test27_reward](./images/test27_reward.jpg)

### test28
Continue test24 with smaller entropyCoef and larger batchSize and maxStep

|name|value|
|----|-----|
|env_num|80|
|#step|10|
|batchSize|80 * 10|
|entropyCoef|0.001|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

No improvement compared to test24:

![test28_24_reward](./images/test28_test24_reward.jpg)

And entropy of test24 seemed more healthy:

![test28_24_entropy](./images/test28_test24_entropy.jpg)

Entropy in above figure has had tail cut. The figure for all updates showed it is decreasi with mild rise and fall:

![test24_entropy_28](./images/test24_entropy.jpg)

### test29
Increase maxStep to check if the agent was short-sighted based on test28

|name|value|
|----|-----|
|env_num|50|
|#step|16|
|batchSize|50 * 16|
|entropyCoef|0.001|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

The training result had been covered by test32

### testtest29
Model of test24 is better, increasing maxStep did not help:

![test29_test24_reward](./images/testtest29_reward.jpg)

###  test30
Increase maxGradNormClip to allow more exceptional samples based on test28

|name|value|
|----|-----|
|env_num|50|
|#step|10|
|batchSize|50 * 10|
|entropyCoef|0.001|
|valueCoef|0.25|
|maxGradNormClip|0.5|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Worse than test24

### test31
Increase entropy and maxStep based on test29

|name|value|
|----|-----|
|env_num|50|
|#step|16|
|batchSize|50 * 16|
|entropyCoef|0.002|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Reward was better than that of test24:

![test31_reward](./images/test31_test24_reward.jpg)

While entropy of test24 seemed better:

![test31_24_entropy](./images/test31_test24_entropy.jpg)

![test31_entropy](./images/test31_entropy.jpg)

### test32
Brute force:

|name|value|
|----|-----|
|env_num|80|
|#step|16|
|batchSize|80 * 16|
|entropyCoef|0.001|
|valueCoef|0.25|
|maxGradNormClip|0.1|
|optimizer|RMS|
|learningRate|7e-4|
|eps|1e-5|
|alpha|0.99|

Seemed better than test31 while still failed to get the optimal policy that average-reward = 21.

![test32_test31_reward](./images/test32_test31_reward.jpg)

![test32_test31_entropy](./images/test32_test31_entropy.jpg)



