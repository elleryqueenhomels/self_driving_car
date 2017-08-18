# Self Driving Car - Agent (AI) - PyTorch

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


CHECKPOINT_PATH = 'last_brain.pth'


# Create the architecture of the Artificial Neural Network
class ANN(nn.Module):

	def __init__(self, input_sz, output_sz, hidden_layer_sizes, hidden_activation=F.relu, output_activation=lambda x: x):
		super(ANN, self).__init__()
		self.input_sz = input_sz
		self.output_sz = output_sz
		self.fh = hidden_activation
		self.fo = output_activation
		# self.layers = []
		# Mi = input_sz
		# hidden_layer_sizes.append(output_sz)
		# for Mo in hidden_layer_sizes:
		# 	layer = nn.Linear(Mi, Mo)
		# 	self.layers.append(layer)
		# 	Mi = Mo

		M = hidden_layer_sizes[0]
		self.fc1 = nn.Linear(input_sz, M)
		self.fc2 = nn.Linear(M, output_sz)

	def forward(self, X):
		# Z = X
		# for layer in self.layers[:-1]:
		# 	Z = self.fh(layer(Z))
		# Q_values = self.fo(self.layers[-1](Z))
		# return Q_values

		Z = self.fh(self.fc1(X))
		Q_values = self.fo(self.fc2(Z))
		return Q_values


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
		return map(lambda x: Variable(torch.cat(x, 0)), zip(*samples))


# Create the Deep Q-Network
class DQN(object):

	def __init__(self, input_sz, output_sz, hidden_layer_sizes, gamma, buffer_capacity=100000, batch_sz=100, learning_rate=1e-3):
		self.gamma = gamma
		self.batch_sz = batch_sz
		self.reward_window = []
		self.model = ANN(input_sz, output_sz, hidden_layer_sizes)
		self.memory = ExperienceReplayBuffer(buffer_capacity)
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

	def get_state(self, signal):
		# signal is a kind of observation of the environment
		return torch.Tensor(signal).float().unsqueeze(0)

	def select_action(self, signal, temperature=75):
		state = self.get_state(signal)
		probs = F.softmax(self.model(Variable(state, volatile=True)) * temperature)
		action = probs.multinomial()
		return action.data[0, 0]

	def learn(self, batch_states, batch_actions, batch_rewards, batch_next_satets):
		outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
		next_outputs = self.model(batch_next_satets).detach().max(1)[0]
		targets = batch_rewards + self.gamma * next_outputs
		# td_loss = F.mse_loss(outputs, targets)
		td_loss = F.smooth_l1_loss(outputs, targets)
		self.optimizer.zero_grad()
		# td_loss.backward(retain_variables=True)
		td_loss.backward(retain_graph=True)
		self.optimizer.step()

	def update(self, signal, action, reward, next_signal):
		state = self.get_state(signal)
		next_state = self.get_state(next_signal)
		event = (state, torch.LongTensor([int(action)]), torch.Tensor([reward]), next_state)
		self.memory.push(event)
		if len(self.memory.buffer) > self.batch_sz:
			batch_states, batch_actions, batch_rewards, batch_next_satets = self.memory.sample(self.batch_sz)
			self.learn(batch_states, batch_actions, batch_rewards, batch_next_satets)
		self.reward_window.append(reward)
		if len(self.reward_window) > 1000:
			self.reward_window.pop(0)

	def score(self):
		if len(self.reward_window) == 0:
			return 0
		return np.mean(self.reward_window)

	def save(self, path=CHECKPOINT_PATH):
		print('=> Saving checkpoint...')
		torch.save({'state_dict': self.model.state_dict(),
					'optimizer' : self.optimizer.state_dict(),
					}, path)
		print('Done! Checkpoint: %s' % path)

	def load(self, path=CHECKPOINT_PATH):
		if os.path.isfile(path):
			print('=> Loading checkpoint...')
			checkpoint = torch.load(path)
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			print('Done!')
		else:
			print('No checkpoint <%s> found...' % path)

