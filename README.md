# Self-Driving Car

A simple Self-Driving Car

- Highly recommend to use IDE (My choice: <b>Spyder</b> 3.1.4)
- By the way, <b>Anaconda</b> already contains Spyder.

The implementation of Double DQN with Prioritized Experience Replay (Proportional Prioritization) is based on van Hasselt et al. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) [2015.12] and Schaul et al. [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) [2016.02].

Here is another interesting paper which demonstrates end-to-end learning for self-driving car by Bojarski et al. [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) [2016.04]. In this paper, researchers train a CNN with a small amount of training data and find that the model is able to learn the entire task of lane and road following without manual decomposition into road or lane marking detection, semantic abstraction, path planning, and control. They evaluate the model with simulation tests and on-road tests.

## Environment
- <b>Python</b> 2.7.13
- <b>NumPy</b> 1.13.1
- <b>Kivy</b> 1.10.0 (For GUI)
- (Optional) <b>Keras</b> 2.0.8
- (Optional) <b>Theano</b> 0.9.0
- (Optional) <b>PyTorch</b> 0.2.0_1
- (Optional) <b>TensorFlow</b> 1.0.* or 1.1.* or 1.2.* or 1.3.*

<b>NOTE</b>: Among Keras, Theano, PyTorch and TensorFlow, we only need to choose one of them, but in the directory 'ddqn_per' we only use Keras.

## ********************* still underdevelopment *********************
