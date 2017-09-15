# Self Driving Car - Agent (AI)
# Further more, using:
# Double DQN with Prioritized Experience Replay (Proportional Prioritization)


import numpy as np
from keras import backend as K
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
from replay_memory import PrioritizedReplayMemory


# ------------------ Constants ------------------
BRAIN_FILE  = 'DDQN_PER.h5'

UPDATE_TARGET_FREQUENCY = 1000

LEARNING_RATE = 0.0025
BATCH_SZ      = 32
GAMMA         = 0.99

MEMORY_START_SIZE = BATCH_SZ
MEMORY_CAPACITY   = 200000
MEMORY_ALPHA      = 0.4
MEMORY_EPS        = 0.01

REWARD_WINDOW_SIZE = 1000


# ------------------ Utilities ------------------
def mse_loss(y_true, y_pred):
    loss = K.square(y_true - y_pred)
    return K.mean(K.sum(loss, axis=1))

def softmax(x):
    y = np.exp(x)
    return y / y.sum()


# ------------------ Classes ------------------
class Brain:

    def __init__(self, input_sz, output_sz, hidden_layer_sizes):
        self.input_sz  = input_sz
        self.output_sz = output_sz
        self.hidden_layer_sizes = hidden_layer_sizes

        self.model = self.create_model()
        self.target_model = self.create_model() # target network
        self.update_target_model()

    def create_model(self):
        x = Input(shape=(self.input_sz, ))

        h = x
        for M in self.hidden_layer_sizes:
            # h = LSTM(units=M)(h)
            h = Dense(units=M, activation='relu')(h)

        y = Dense(units=self.output_sz, activation=None)(h)

        optim = Adam(lr=LEARNING_RATE)
        # optim = RMSprop(lr=LEARNING_RATE)

        model = Model(inputs=x, outputs=y)
        model.compile(loss=mse_loss, optimizer=optim)

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, states, target=False):
        if target:
            return self.target_model.predict(states)
        else:
            return self.model.predict(states)

    def predict_one(self, state, target=False):
        states = np.expand_dims(state, axis=0)
        return self.predict(states, target)[0]

    def get_td_error(self, state, action, reward, next_state):
        best_action = np.argmax(self.predict_one(next_state))
        next_return = self.predict_one(next_state, target=True)[best_action]
        target      = reward + GAMMA * next_return

        prediction  = self.predict_one(state)[action]
        td_error    = abs(target - prediction)

        return td_error

    def train(self, states, actions, rewards, next_states, epochs=1, verbose=0):
        batch_range  = np.arange(len(actions))

        best_actions = np.argmax(self.predict(next_states), axis=1)
        next_returns = self.predict(next_states, target=True)[batch_range, best_actions]
        targets      = rewards + GAMMA * next_returns

        outputs      = self.predict(states)
        predictions  = outputs[batch_range, actions]
        td_errors    = np.abs(targets - predictions)

        x = states
        y = outputs
        y[batch_range, actions] = targets

        self.model.fit(x, y, batch_size=None, epochs=epochs, verbose=verbose)

        return td_errors

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)
        self.update_target_model()


class Agent:

    def __init__(self, input_sz, output_sz, hidden_layer_sizes):
        self.input_sz  = input_sz
        self.output_sz = output_sz

        self.brain  = Brain(input_sz, output_sz, hidden_layer_sizes)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY, alpha=MEMORY_ALPHA, eps=MEMORY_EPS)

        self.training_steps = 0
        self.reward_window  = []

    def select_action(self, state, temperature):
        value = self.brain.predict_one(state)
        probs = softmax(value * temperature)
        return np.random.choice(self.output_sz, p=probs)

    def train(self, state, action, reward, next_state):
        event = (state, action, reward, next_state)
        td_error = self.brain.get_td_error(state, action, reward, next_state)
        self.memory.push(event, td_error)

        self.reward_window.append(reward)
        if len(self.reward_window) > REWARD_WINDOW_SIZE:
            self.reward_window.pop(0)

        if self.memory.current_length() > MEMORY_START_SIZE:
            if self.training_steps % UPDATE_TARGET_FREQUENCY == 0:
                self.brain.update_target_model()

            self.training_steps += 1

            # train the brain with prioritized experience replay
            samples, indices, priorities = self.memory.sample(BATCH_SZ)
            states, actions, rewards, next_states = samples

            td_errors = self.brain.train(states, actions, rewards, next_states)
            self.memory.update(indices, td_errors)

    def score(self):
        if len(self.reward_window) == 0:
            return 0.0
        return np.mean(self.reward_window)

    def save_brain(self, path):
        self.brain.save(path)

    def load_brain(self, path):
        self.brain.load(path)

