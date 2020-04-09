import os
import random
import utils
from collections import deque
import config
import csv
import numpy as np
import pandas

class Agent():
    def __init__(self):

        self.weight_backup = ".//resources//agent//dqn_weight.h5"
        self.memory = deque(maxlen=2000)

        self.qtable_columns, self.actions = self.load_actions_lexicon(config.AGENT_ACTION_SPACE)
        self.action_encoder = None

        self.qtable = pandas.DataFrame(0, index=[], columns=self.qtable_columns)

        self.learning_rate = config.LEARNING_RATE
        self.gamma = config.GAMMA
        self.exploration_rate = config.EPSILON
        self.exploration_min = config.EPSILON_MIN
        self.exploration_decay = config.EPSILON_DECAY
        self.model = None

    def load_actions_lexicon(self, path):
        with open(path) as f:
            list_actions = []
            list_columns = []
            for line in f:
                line_qtable_columns = line.replace('\n','')
                line_input = line.replace('\n','').split(',')
                list_columns.append(line_qtable_columns)
                list_actions.append(line_input)
        return list_columns, list_actions

    def save_model(self):
        self.model.save(self.weight_backup)


    def update_qtables(self, prev_state, current_state, agent_action, reward):
        action = agent_action['ack_cs'] + "," + agent_action['intent'] + "," + agent_action['cs']
        if str(prev_state) in self.qtable.index:
            if str(current_state) not in self.qtable.index:
                self.qtable.loc[str(current_state)] = 0
            discounted_reward = (1 - self.learning_rate) * self.qtable.at[str(prev_state), action] + self.learning_rate * (
                        reward + self.gamma * np.max(self.qtable.loc[str(current_state), :]))
            self.qtable.at[str(prev_state), action] = discounted_reward
        else:
            self.qtable.loc[str(prev_state)] = 0
            if str(current_state) not in self.qtable.index:
                self.qtable.loc[str(current_state)] = 0
            self.qtable.at[str(prev_state), action] = (1 - self.learning_rate) * self.qtable.at[str(prev_state), action] + self.learning_rate * (reward + self.gamma * np.max(self.qtable.loc[str(current_state), :]))


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
            agent_action = self.next_best(state)
        else:
            agent_action = random.choice(self.actions)
        entity_type = self.pick_slot(state, agent_action)
        new_msg = self.msg_to_json(agent_action, entity_type)
        return new_msg

    def next_best(self, state):
        current_state = self.qtable.loc[str(state)]
        action = current_state.idxmax()
        return action

    def pick_slot(self, state, agent_action):
        if 'request' in agent_action[1]:
            if "genre" not in state.state["slots_filled"]:
                return "genre"
            if "actor" not in state.state["slots_filled"]:
                return "actor"
            if "director" not in state.state["slots_filled"]:
                return "director"
            else:
                return random.choice(["genre","actor","director"])
        elif 'introduce' in agent_action[1]:
            if "role" not in state.state["introduce_slots"]:
                return "role"
            if "last_movie" not in state.state["introduce_slots"]:
                return "last_movie"
            if "reason_like" not in state.state["introduce_slots"]:
                return "reason_like"
            else:
                return random.choice(["role","last_movie","reason_like"])
        elif 'inform' in agent_action[1]:
            return "movie"
        else:
            return ""

    def msg_to_json(self, intention, entity_type):
        frame = {'intent': intention[1], 'entity_type': entity_type, 'ack_cs': intention[0], 'cs': intention[2]}
        return frame