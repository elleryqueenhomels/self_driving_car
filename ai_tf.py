# Self Driving Car - Agent (AI) - TensorFlow

import os
import random
import numpy as np
import tensorflow as tf


CHECKPOINT_PATH = 'last_brain.npz'


# Create the hidden layer class for Neural Network
class HiddenLayer(object):

	def __init__(self, Mi, Mo, activation=tf.nn.relu, use_bias=True):
		self.W = tf.Variable(tf.truncated_normal(shape=(Mi, Mo)))
		self.use_bias = use_bias
		self.params = [self.W]
		if use_bias:
			self.b = tf.Variable(np.zeros(Mo, dtype=np.float32))
			self.params.append(self.b)
		self.f = activation

	def forward(self, X):
		if self.use_bias:
			Z = tf.matmul(X, self.W) + self.b
		else:
			Z = tf.matmul(X, self.W)
		return self.f(Z)


# Create the Experience Replay Buffer
class ExperienceReplayBuffer(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []

	def push(self, event):
		self.buffer.append(event)
		if len(self.buffer) > self.capacity:
			self.buffer.pop(0)

	def sample(self, batch_sz):
		samples = random.sample(self.buffer, batch_sz)
		return map(np.array, zip(*samples))


# Create the Deep Q-Network
class DQN(object):

	def __init__(self, input_sz, output_sz, hidden_layer_sizes, gamma, buffer_capacity=100000, batch_sz=100, learning_rate=1e-3):
		self.gamma = gamma
		self.batch_sz = batch_sz
		self.reward_window = []
		self.memory = ExperienceReplayBuffer(buffer_capacity)

		# create the graph
		self.layers = []
		Mi = input_sz
		for Mo in hidden_layer_sizes:
			layer = HiddenLayer(Mi, Mo, activation=tf.nn.relu)
			self.layers.append(layer)
			Mi = Mo

		# final layer
		layer = HiddenLayer(Mi, output_sz, activation=lambda x: x)
		self.layers.append(layer)

		# collect the params
		self.params = []
		for layer in self.layers:
			self.params += layer.params

		# inputs and targets
		self.states = tf.placeholder(tf.float32, shape=(None, input_sz), name='states')
		self.actions = tf.placeholder(tf.int32, shape=(None, ), name='actions')
		self.targets = tf.placeholder(tf.float32, shape=(None, ), name='targets')

		# outputs and cost
		Z = self.states
		for layer in self.layers:
			Z = layer.forward(Z)
		Q_values = Z
		self.predict_op = Q_values

		selected_action_values = tf.reduce_sum(
			Q_values * tf.one_hot(self.actions, output_sz),
			reduction_indices=[1]
		)

		# An alternative method to calculate selected_action_values:
		# we would like to do this, but it doesn't work in TensorFlow:
		# selected_action_values = Q_values[tf.range(batch_sz), self.actions]
		# instead we do:
		# indices = tf.range(tf.shape(Q_values)[0]) * tf.shape(Q_values)[1] + self.actions
		# selected_action_values = tf.gather(
		# 	tf.reshape(Q_values, [-1]),
		# 	indices
		# )

		# cost = tf.reduce_mean(tf.square(self.targets - selected_action_values)) # may use tf.reduce_sum()
		cost = tf.nn.l2_loss(self.targets - selected_action_values)

		self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
		# self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.0, epsilon=1e-6).minimize(cost)

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())

	def set_session(self, session):
		self.session = session

	def predict(self, states):
		states = np.atleast_2d(states)
		return self.session.run(self.predict_op, feed_dict={self.states: states})

	def select_action(self, state, temperature=50):
		state = np.atleast_2d(state)
		probs_op = tf.nn.softmax(self.predict_op * temperature)
		probs = self.session.run(probs_op, feed_dict={self.states: state})[0]
		return np.random.choice(len(probs), p=probs)

	def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
		next_outputs = self.predict(batch_next_states).max(axis=1)
		targets = batch_rewards + self.gamma * next_outputs
		self.session.run(self.train_op, feed_dict={self.states: batch_states, self.actions: batch_actions, self.targets: targets})

	def update(self, state, action, reward, next_state):
		event = (state, action, reward, next_state)
		self.memory.push(event)
		if len(self.memory.buffer) > self.batch_sz:
			batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(self.batch_sz)
			self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
		self.reward_window.append(reward)
		if len(self.reward_window) > 1000:
			self.reward_window.pop(0)

	def score(self):
		if len(self.reward_window) == 0:
			return 0
		return np.mean(self.reward_window)

	def save(self, path=CHECKPOINT_PATH):
		print('=> Saving checkpoint...')
		params = [p.eval(self.session) for p in self.params]
		np.savez(path, *params)
		print('Done! Checkpoint: %s' % path)

	def load(self, path=CHECKPOINT_PATH):
		if os.path.isfile(path):
			print('=> Loading checkpoint...')
			npz = np.load(path)
			ops = []
			for i in range(len(self.params)):
				op = self.params[i].assign(npz['arr_%d' % i])
				ops.append(op)
			self.session.run(ops)
			print('Done!')
		else:
			print('No checkpoint <%s> found...' % path)

