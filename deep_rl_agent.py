import os
import random
from collections import deque
import config
import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent():
    def __init__(self):

        self.weight_backup = ".//resources//agent//dqn_weight.h5"
        self.memory = deque(maxlen=2000)

        self.actions = config.AGENT_ACTIONS
        self.social_qtable = pandas.DataFrame(0, index=[], columns=config.CS_LABELS)
        self.ack_qtable = pandas.DataFrame(0, index=[], columns=config.CS_LABELS)

        self.learning_rate = config.LEARNING_RATE
        self.gamma = config.GAMMA
        self.exploration_rate = config.EPSILON
        self.exploration_min = config.EPSILON_MIN
        self.exploration_decay = config.EPSILON_DECAY
        self.model = self.build_DQN_model()

    def build_DQN_model(self):
        # Neural Net for Deep-Q learning Model
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

    def vectorize_action(self, action):
        action_vector = [0,0,0,0,0,0,0]
        if "greeting" in action:
            action_vector[0] = 1
        if "request(last_movie)" in action:
            action_vector[1] = 1
        if "request(cast)" in action:
            action_vector[2] = 1
        if "request(crew)" in action:
            action_vector[3] = 1
        if "request(genre)" in action:
            action_vector[4] = 1
        if "inform(movie)" in action:
            action_vector[5] = 1
        if "goodbye" in action:
            action_vector[6] = 1
        action_vector = np.asarray(action_vector)
        action_vector = action_vector[np.newaxis, :]
        return action_vector

    def devectorize_action(self, action):
        action_vector = [0,0,0,0,0,0,0]
        if action == 0:
            action_name = "greeting"
        if action == 1:
            action_name = "request(last_movie)"
        if action == 2:
            action_name = "request(cast)"
        if action == 3:
            action_name = "request(crew)"
        if action == 4:
            action_name = "request(genre)"
        if action == 5:
            action_name = "inform(movie)"
        if action == 6:
            action_name = "goodbye"
        return action_name


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

    def pick_best_cs(self, state, user_action):
        cs_row = []
        cs_row.append(state["user_social_type"])
        cs_row.append(user_action["cs"])
        current_state = self.social_qtable.loc[str(cs_row)]
        agent_cs = current_state.idxmax()
        return agent_cs

    def update_qtables(self, prev_state, current_state, agent_action, agent_previous_action, user_action, user_previous_action, reward):

        previous_cs_row = []
        previous_cs_row.append(current_state["user_social_type"])
        previous_cs_row.append(user_previous_action["cs"])
        cs_row = []
        cs_row.append(current_state["user_social_type"])
        cs_row.append(user_action["cs"])
            # update social qtable
        if str(previous_cs_row) in self.social_qtable.index:
            if str(cs_row) not in self.social_qtable.index:
                self.social_qtable.loc[str(cs_row)] = 0
            self.social_qtable.at[str(previous_cs_row), agent_action['cs']] = (1 - self.learning_rate) * self.social_qtable.at[
                str(previous_cs_row), agent_action['cs']] + self.learning_rate * (reward + self.gamma * np.max(
                self.social_qtable.loc[str(cs_row), :]))
        else:
            self.social_qtable.loc[str(previous_cs_row)] = 0
            if str(cs_row) not in self.social_qtable.index:
                self.social_qtable.loc[str(cs_row)] = 0
            self.social_qtable.at[str(previous_cs_row), agent_action['cs']] = (1 - self.learning_rate) * self.social_qtable.at[
                str(previous_cs_row), agent_action['cs']] + self.learning_rate * (reward + self.gamma * np.max(
                self.social_qtable.loc[str(cs_row), :]))