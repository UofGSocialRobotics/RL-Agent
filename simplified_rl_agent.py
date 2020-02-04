import json
import random
import config
import pandas
import numpy as np


class Agent():
    def __init__(self):

        self.actions = ["greeting", "request(last_movie)", "request(cast)", "request(crew)", "request(genre)", "inform(movie)", "goodbye"]
        self.task_qtable = pandas.DataFrame(0, index=[], columns=self.actions)
        self.social_qtable = pandas.DataFrame(0, index=[], columns=config.CS_LABELS)
        self.ack_qtable = pandas.DataFrame(0, index=[], columns=config.CS_LABELS)

        self.epsilon = 1.0  # Greed 100%
        self.epsilon_min = 0.005  # Minimum greed 0.05%
        self.epsilon_decay = 0.99993  # Decay multiplied with epsilon after each episode
        self.learning_rate = 0.65
        self.gamma = 0.65


    def next(self):
        next_state = random.choice(self.actions)

        agent_cs = self.pick_cs()
        ack_cs = self.pick_cs()
        new_msg = self.msg_to_json(next_state, ack_cs, agent_cs)

        return new_msg

    def next_best(self, state, user_action):
        current_state = self.task_qtable.loc[str(state)]
        action = current_state.idxmax()
        #Todo Get rid of entity in state for qtable
        agent_cs = self.pick_best_cs(state, user_action)
        ack_cs = self.pick_best_cs(state, user_action)
        new_msg = self.msg_to_json(action, ack_cs, agent_cs)

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
        # update task qtable
        if str(prev_state) in self.task_qtable.index:
            if str(current_state) not in self.task_qtable.index:
                self.task_qtable.loc[str(current_state)] = 0
            self.task_qtable.at[str(prev_state), agent_action['intent']] = (1 - self.learning_rate) * self.task_qtable.at[
                str(prev_state), agent_action['intent']] + self.learning_rate * (reward + self.gamma * np.max(
                self.task_qtable.loc[str(current_state), :]))
        else:
            self.task_qtable.loc[str(prev_state)] = 0
            if str(current_state) not in self.task_qtable.index:
                self.task_qtable.loc[str(current_state)] = 0
            self.task_qtable.at[str(prev_state), agent_action['intent']] = (1 - self.learning_rate) * self.task_qtable.at[
                str(prev_state), agent_action['intent']] + self.learning_rate * (reward + self.gamma * np.max(
                self.task_qtable.loc[str(current_state), :]))


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