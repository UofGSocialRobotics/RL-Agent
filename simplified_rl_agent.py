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

    def next_best(self, state):
        current_state = self.task_qtable.loc[str(state)]
        action = current_state.idxmax()

        agent_cs = self.pick_best_cs(state)
        ack_cs = self.pick_best_cs(state)
        new_msg = self.msg_to_json(action, ack_cs, agent_cs)

        return new_msg

    def msg_to_json(self, intention, ack_cs, cs):
        frame = {'intent': intention, 'ack_cs': ack_cs, 'cs': cs}
        return frame

    def pick_cs(self):
        agent_cs = random.choice(config.CS_LABELS)
        return agent_cs

    def pick_best_cs(self, state):
        current_state = self.social_qtable.loc[str(state)]
        agent_cs = current_state.idxmax()
        return agent_cs

    def update_qtables(self, prev_state, current_state, agent_action, reward):
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

        # cs_row = []
        # cs_row.append(prev_state["user_social_type"])
        # cs_row.append(prev_state)
        # # update social qtable
        # if str(state) in agent.social_qtable.index:
        #     if str(dst.state) not in agent.social_qtable.index:
        #         agent.social_qtable.loc[str(dst.state)] = 0
        #     agent.social_qtable.at[str(state), agent_action['cs']] = (1 - agent.learning_rate) * agent.social_qtable.at[
        #         str(state), agent_action['cs']] + agent.learning_rate * (reward + agent.gamma * np.max(
        #         agent.social_qtable.loc[str(dst.state), :]))
        # else:
        #     agent.social_qtable.loc[str(state)] = 0
        #     if str(dst.state) not in agent.social_qtable.index:
        #         agent.social_qtable.loc[str(dst.state)] = 0
        #     agent.social_qtable.at[str(state), agent_action['cs']] = (1 - agent.learning_rate) * agent.social_qtable.at[
        #         str(state), agent_action['cs']] + agent.learning_rate * (reward + agent.gamma * np.max(
        #         agent.social_qtable.loc[str(dst.state), :]))
