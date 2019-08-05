import random
import config
import utils
import pandas


class Agent():
    def __init__(self):

        # Do we store the users preferences in a user model?
        self.store_pref = True

        self.actions = ["greeting", "request(last_movie)", "request(genre)", "inform(movie)", "goodbye"]

        self.cs_qtable = pandas.DataFrame(0, index=[], columns=self.actions)
        print(self.cs_qtable)


    def next(self, state):
        if state["turns"] == 0:
            next_state = "greeting"
        else:
            next_state = random.choice(self.actions)

        # self.actions.remove(next_state)

        agent_cs = self.pick_cs()
        ack_cs = self.pick_cs()
        new_msg = self.msg_to_json(next_state, ack_cs, agent_cs)

        return new_msg

    def msg_to_json(self, intention, ack_cs, cs):
        frame = {'intent': intention, 'ack_cs': ack_cs, 'cs': cs}
        return frame

    def pick_cs(self):
        agent_cs = random.choice(config.CS_LABELS)
        return agent_cs