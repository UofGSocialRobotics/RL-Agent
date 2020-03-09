import os
import random
import utils
from collections import deque
import config
import pandas
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent():
    def __init__(self):

        self.weight_backup = ".//resources//agent//dqn_weight.h5"
        self.memory = deque(maxlen=2000)

        self.actions = config.AGENT_ACTIONS
        self.action_encoder = None

        self.learning_rate = config.LEARNING_RATE
        self.gamma = config.GAMMA
        self.exploration_rate = config.EPSILON
        self.exploration_min = config.EPSILON_MIN
        self.exploration_decay = config.EPSILON_DECAY
        #self.model = self.build_DQN_model()

    def build_DQN_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=config.DQN_STATE_SPACE, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(len(self.actions), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            self.model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_action_space_encoder(self):
        action_space = []
        with open(config.AGENT_ACTION_SPACE, mode='rt') as csv_file:
            interaction = csv.reader(csv_file, delimiter=',')
            for row in interaction:
                action_space.append(row)
        action_encoder = utils.encode(action_space)
        return action_encoder

    def next(self, state):
        if random.uniform(0, 1) > config.EPSILON:
            vectorized_state = state.vectorize()
            act_values = self.model.predict(vectorized_state)
            action = np.argmax(act_values[0])
            agent_action = self.devectorize_action(action)
        else:
            agent_action = random.choice(self.actions)

        agent_cs = self.pick_cs()
        ack_cs = self.pick_cs()
        new_msg = self.msg_to_json(agent_action, ack_cs, agent_cs)
        return new_msg

    def msg_to_json(self, intention, ack_cs, cs):
        frame = {'intent': intention, 'ack_cs': ack_cs, 'cs': cs}
        return frame

    def pick_cs(self):
        agent_cs = random.choice(config.CS_LABELS)
        return agent_cs