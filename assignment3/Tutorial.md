# 实验二： DQN for Atari game
### Atari Game
![Atari](other/atari.png)
### 需要安装的包
	1. gym
	2. numpy
	3. tensorflow or tensorflow-gpu   推荐安装 1.14.0     2.0+ 版本 会有一些bug
		.注意tensorflow 和  tensorflow-gpu的安装 需要对应 python3.6、 cuda、cudnn的版本
		.查看 cuda 版本  cat /usr/local/cuda/version.txt    推荐版本  cuda 10
		.查看cudnn 版本  cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2    推荐版本 7.4
						升级参考链接   https://blog.csdn.net/zong596568821xp/article/details/86098833
						安装参考链接	https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#prerequisites
		.版本查询网址可以参考    https://blog.csdn.net/K1052176873/article/details/114526086
		.具体安装可以参考     https://tf.wiki/zh_hans/basic/installation.html
		
	4. 可能还要更多。。。

### 在dqn.py中找需要填的代码
	1. 共7个代码框，每个代码框约1-4行代码即可，总共20行代码左右即可。具体看每个代码框需要的代码，有相应的提示。
	2. 不会tensorflow没关系，只需要调用lib/dqn_utils.py下函数即可。
	3. 不太需要改神经网络结构，用的是https://www.nature.com/articles/nature14236框架。

### dqn algorithm
![ dqn ]( other/dqn.png )

#### CODE 1 : Populate replay memory
	先保存一定数量的episodes在经验池中，再进行训练
	这个参数是 replay_memory_init_size
	Hints: 在lib/dqn_utils.py下找函数 populate_replay_buffer
	（约1行代码）

#### CODE 2 : 	Target network update
	每隔一定的时间间隔要把 update network 拷贝到target network里去
	hints:在lib/dqn_utils.py下找函数copy_model_parameters
	（约1行代码）
	
#### CODE 3: Take a step in the environment
	hints 1： 这里注意下对state的处理
	hints 2: 如果不会写，可以参考下函数 populate_replay_buffer()
	（约几行代码）
#### CODE 4: Save transition to replay memory
	添加一条experience到经验池中
	hints: 可看下函数populate_replay_buffer
	（约1-2行代码）
	
#### CODE 5: Sample a minibatch from the replay memory
	从经验池中抽取batch_size条经验，用于网络更新
	deepmind论文中batch_size=32
	hints: 使用函数 random.sample( replay_memory, batch_size )
	（约1-2行代码）

#### CODE 6: use minibatch sample to calculate q values and targets
	计算Q(s, a)值 和 r+max Q(s', a')
	hints 1: 使用函数 q_estimator.predict 计算Q(s, a)
	hints 2: 使用函数 target_estimator.predict计算Q(s', a')
	（有兴趣可以实现下 double-q learning,只需要在这里改下就行了）
	（约几行代码）

#### CODE 7: Perform gradient descent update
	更新update network
	hints : 函数 'q_estimator.update'
	（1行代码）


### **最后，提示下，代码量少，但跑程序时间可能30小时+，tensorflow-gpu版本会快好几倍**


### 参考文献
 [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

