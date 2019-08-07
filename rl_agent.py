import random
import config
import utils
import pandas


class Agent():
    def __init__(self):

        if config.GENERATE_SENTENCE:
            self.sentenceDB = utils.load_agent_sentence_model(config.AGENT_SENTENCES)
            self.ackDB = utils.load_agent_ack_model(config.AGENT_ACKS)

        if config.GENERATE_VOICE:
            self.engine = utils.set_voice_engine("A", config.AGENT_VOICE)

        self.actions = self.load_actions_list(config.AGENT_ACTIONS)
        self.cs_qtable = pandas.DataFrame(0, index=[], columns=self.actions)

        self.epsilon = 1.0  # Greed 100%
        self.epsilon_min = 0.005  # Minimum greed 0.05%
        self.epsilon_decay = 0.99993  # Decay multiplied with epsilon after each episode
        self.learning_rate = 0.65
        self.gamma = 0.65

    # Parse the model.csv file and transform that into a dict of Nodes representing the scenario
    def load_actions_list(self, path):
        actions = []
        with open(path) as f:
            for line in f:
                if "#" not in line:
                    actions.append(line.replace("\n",""))
        return actions

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
