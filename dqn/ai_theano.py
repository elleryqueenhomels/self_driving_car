# Self Driving Car - Agent (AI) - Theano

import os
import random
import numpy as np
import theano
import theano.tensor as T


CHECKPOINT_PATH = 'last_brain_theano.npz'


# Create the hidden layer class for Neural Network
class HiddenLayer(object):

	def __init__(self, Mi, Mo, activation=T.nnet.relu, use_bias=True):
		W0 = np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)
		self.W = theano.shared(W0.astype(np.float32))
		self.use_bias = use_bias
		self.params = [self.W]
		if use_bias:
			b0 = np.zeros(Mo, dtype=np.float32)
			self.b = theano.shared(b0)
			self.params.append(self.b)
		self.f = activation

	def forward(self, X):
		if self.use_bias:
			Z = X.dot(self.W) + self.b
		else:
			Z = X.dot(self.W)
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

	def __init__(self, input_sz, output_sz, hidden_layer_sizes, gamma, buffer_capacity=100000, batch_sz=100, learning_rate=1e-3, decay=0.999, momentum=0, eps=1e-10):
		lr = np.float32(learning_rate)
		mu = np.float32(momentum)
		decay = np.float32(decay)
		eps = np.float32(eps)
		one = np.float32(1)

		self.gamma = gamma
		self.batch_sz = batch_sz
		self.reward_window = []
		self.memory = ExperienceReplayBuffer(buffer_capacity)

		# create the graph
		layers = []
		Mi = input_sz
		for Mo in hidden_layer_sizes:
			layer = HiddenLayer(Mi, Mo, activation=T.nnet.relu)
			layers.append(layer)
			Mi = Mo

		# final layer
		layer = HiddenLayer(Mi, output_sz, activation=lambda x: x)
		layers.append(layer)

		# collect the params
		self.params = []
		for layer in layers:
			self.params += layer.params

		# inputs and targets
		states = T.fmatrix('states')
		actions = T.ivector('actions')
		targets = T.fvector('targets')

		# outputs and cost
		Z = states
		for layer in layers:
			Z = layer.forward(Z)
		Q_values = Z

		selected_action_values = Q_values[T.arange(actions.shape[0]), actions]
		cost = T.sum((targets - selected_action_values)**2) # may use T.sum() or T.mean()

		# for sample action
		temperature = T.fscalar('temperature')
		probs = T.nnet.softmax(Q_values * temperature)

		# create train function
		grads = T.grad(cost, self.params)
		self.caches = [theano.shared(np.ones_like(p.get_value(), dtype=np.float32)*np.float32(0.1)) for p in self.params]
		self.velocities = [theano.shared(p.get_value()*np.float32(0)) for p in self.params]

		c_update = [(c, decay*c + (one - decay)*g*g) for c, g in zip(self.caches, grads)]

		# optional to use momentum
		if mu > 0:
			v_update = [(v, mu*v - lr*g / T.sqrt(c + eps)) for v, c, g in zip(self.velocities, self.caches, grads)]
			p_update = [(p, p + v) for p, v in zip(self.params, self.velocities)]
			updates  = c_update + v_update + p_update
		else:
			p_update = [(p, p - lr*g / T.sqrt(c + eps)) for p, c, g in zip(self.params, self.caches, grads)]
			updates  = c_update + p_update

		# compile functions
		self.predict_op = theano.function(
			inputs=[states],
			outputs=Q_values,
			allow_input_downcast=True
		)

		self.probs_op = theano.function(
			inputs=[states, temperature],
			outputs=probs,
			allow_input_downcast=True
		)

		self.train_op = theano.function(
			inputs=[states, actions, targets],
			updates=updates,
			allow_input_downcast=True
		)

	def predict(self, states):
		states = np.atleast_2d(states)
		return self.predict_op(states)

	def select_action(self, state, temperature=50):
		state = np.atleast_2d(state)
		probs = self.probs_op(state, temperature)[0]
		return np.random.choice(len(probs), p=probs)

	def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
		next_outputs = self.predict(batch_next_states).max(axis=1)
		targets = batch_rewards + self.gamma * next_outputs
		self.train_op(batch_states, batch_actions, targets)

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
		params = [p.get_value() for p in self.params]
		caches = [c.get_value() for c in self.caches]
		velocities = [v.get_value() for v in self.velocities]
		npz = params + caches + velocities
		np.savez(path, *npz)
		print('Done! Checkpoint: %s' % path)

	def load(self, path=CHECKPOINT_PATH):
		if os.path.isfile(path):
			print('=> Loading checkpoint...')
			npz = np.load(path)
			lp, lc, lv = len(self.params), len(self.caches), len(self.velocities)
			for i in range(lp):
				self.params[i].set_value(npz['arr_%d' % i])
			for i in range(lc):
				self.caches[i].set_value(npz['arr_%d' % (lp + i)])
			for i in range(lv):
				self.velocities[i].set_value(npz['arr_%d' % (lp + lc + i)])
			print('Done!')
		else:
			print('No checkpoint <%s> found...' % path)

