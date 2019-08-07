import random
import config
import pandas


class Agent():
    def __init__(self):

        self.actions = ["greeting", "request(last_movie)", "request(actor)", "request(director)", "request(genre)", "inform(movie)", "goodbye"]
        self.cs_qtable = pandas.DataFrame(0, index=[], columns=self.actions)

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
        current_state = self.cs_qtable.loc[str(state)]
        action = current_state.idxmax()

        agent_cs = self.pick_cs()
        ack_cs = self.pick_cs()
        new_msg = self.msg_to_json(action, ack_cs, agent_cs)

        return new_msg

    def msg_to_json(self, intention, ack_cs, cs):
        frame = {'intent': intention, 'ack_cs': ack_cs, 'cs': cs}
        return frame

    def pick_cs(self):
        agent_cs = random.choice(config.CS_LABELS)
        return agent_cs