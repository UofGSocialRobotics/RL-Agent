import os
import random
import utils
from collections import deque
import config
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent():
    def __init__(self):

        self.weight_backup = ".//resources//agent//dqn_weight.h5"
        self.memory = deque(maxlen=2000)

        self.actions = self.load_actions_lexicon(config.AGENT_ACTION_SPACE)
        self.action_encoder = None

        self.learning_rate = config.LEARNING_RATE
        self.gamma = config.GAMMA
        self.exploration_rate = config.EPSILON
        self.exploration_min = config.EPSILON_MIN
        self.exploration_decay = config.EPSILON_DECAY
        self.model = None

    def load_actions_lexicon(self, path):
        with open(path) as f:
            list_actions = []
            for line in f:
                line_input = line.replace('\n','').split(',')
                list_actions.append(line_input)
        return list_actions

    def build_DQN_model(self, state_space_length, action_space_length):
        # model = Sequential()
        # model.add(Flatten(input_shape=(1,state_space_length)))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dense(action_space_length))
        # model.add(Activation('linear'))
        # print(model.summary())

        # input = Input(shape=(1,state_space_length))
        # x = Flatten()(input)
        # x = Dense(16, activation='relu')(x)
        # x = Dense(16, activation='relu')(x)
        # output = Dense(action_space_length, activation='linear')(x)
        # model = Model(inputs=input, outputs=output)
        # print(model.summary())
        # return model
        model = Sequential()
        model.add(Dense(24, input_dim=state_space_length, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(action_space_length, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        #print(model.summary())

        # input = Input(shape=(state_space_length,))
        # hidden = Dense(24, activation='relu')(input)
        # hidden2 = Dense(12, activation='relu')(hidden)
        # output = Dense(action_space_length, activation='softmax')(hidden2)
        # model = Model(inputs=input, outputs=output)
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        # print(model.summary())

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

        cpt = 0
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            cpt += 1
            target = reward
            if not done:
                #print("In Replay ", cpt)
                # print(next_state)
                #next_state = np.array([next_state])
                next_state = np.asarray([next_state], dtype=int)
                # print(type(next_state))
                # print(next_state)
                target = reward + self.gamma * np.amax(self.model.predict(next_state))
            state = np.array([state])
            target_f = self.model.predict(state)
            action = action.toarray().tolist()[0]
            action = np.asarray(action, dtype=int)
            #print(str(action))
            #print(type(action))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


        # for sample in samples:
        #     state, action, reward, new_state, done = sample
        #     target = self.target_model.predict(state)
        #     if done:
        #         target[0][action] = reward
        #     else:
        #         Q_future = max(self.target_model.predict(new_state)[0])
        #         target[0][action] = reward + Q_future * self.gamma
        #     self.model.fit(state, target, epochs=1, verbose=0)


    def get_action_space_encoder(self):
        action_space = []
        with open(config.AGENT_ACTION_SPACE, mode='rt') as csv_file:
            interaction = csv.reader(csv_file, delimiter=',')
            for row in interaction:
                action_space.append(row)
        action_encoder = utils.encode(action_space)
        encoded_action_space = action_encoder.transform(action_space)
        return action_encoder, encoded_action_space

    def next(self, state):
        if random.uniform(0, 1) > config.EPSILON:
            vectorized_state = state.vectorize()
            act_values = self.model.predict(vectorized_state)
            action = np.argmax(act_values[0])
            agent_action = self.devectorize_action(action)
        else:
            agent_action = random.choice(self.actions)

        new_msg = self.msg_to_json(agent_action)
        #print("agent_action")
        return new_msg

    def msg_to_json(self, intention):
        frame = {'intent': intention[1], 'entity_type': '', 'ack_cs': intention[0], 'cs': intention[2]}
        return frame