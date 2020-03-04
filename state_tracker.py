import ml_models
import numpy as np


class DialogState():

    def __init__(self):
        self.set_initial_state()

    def set_initial_state(self):
        self.full_dialog = {}
        self.dialog_done = False
        self.turns = 0
        self.accepted_recos = 0
        self.delivered_recos = 0
        self.state = {}

        self.rec_I_user = [0, 0, 0, 0, 0]
        self.rec_I_agent = [0, 0, 0, 0, 0]
        self.rec_P_agent = 0

        self.state["slots_filled"] = []
        self.state["current_user_action"] = {}
        self.state["previous_user_action"] = {}
        self.state["previous_agent_action"] = {}
        self.state["current_agent_action"] = {}

    def update_state(self, agent_action, user_action, agent_previous_action, user_previous_action):
        self.turns += 1.0

        #update slots
        if "request" in agent_action['intent'] and "director" in agent_action['entity_type'] and "director" not in self.state["slots_filled"]:
            self.state["slots_filled"].append("director")
        if "request" in agent_action['intent'] and "actor" in agent_action['entity_type'] and "actor" not in self.state["slots_filled"]:
            self.state["slots_filled"].append("actor")
        if "request" in agent_action['intent'] and "genre" in agent_action['entity_type'] and "genre" not in self.state["slots_filled"]:
            self.state["slots_filled"].append("genre")
        if "last_movie" in agent_action['intent'] and "last_movie" not in self.state["slots_filled"]:
            self.state["slots_filled"].append("last_movie")
        if "inform" in user_action['intent']:
            if "director" in user_action['entity_type'] and "director" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("director")
            if "actor" in user_action['entity_type'] and "actor" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("actor")
            if "genre" in user_action['entity_type'] and "genre" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("genre")
        self.state["slots_filled"].sort()

        #update agent and user actions
        self.state["current_user_action"] = user_action
        self.state["previous_user_action"] = user_previous_action
        self.state["current_agent_action"] = agent_action
        self.state["previous_agent_action"] = agent_previous_action

        print(self.state["current_agent_action"])
        print(self.state["current_user_action"])

        #update recos
        if "inform" in agent_action['intent'] and "movie" in agent_action['entity_type']:
                    self.delivered_recos += 1
        if "inform" in agent_action['intent']:
            if "yes" in user_action['intent'] or "affirm" in user_action['intent']:
                self.accepted_recos += 1

        if "bye" in agent_action['intent'] or "bye" in user_action['intent']:
            self.dialog_done = True
        self.append_data_from_simulation(agent_action, user_action, agent_previous_action, user_previous_action)

    def append_data_from_simulation(self, agent_action, user_action, agent_previous_action, user_previous_action):
        if "NONE" in user_action['cs'] and "NONE" in agent_action['cs']:
            self.rec_P_agent += 1
        if "NONE" not in user_action['cs']:
            ml_models.count(agent_action['cs'], self.rec_I_user)
        if "start" not in agent_previous_action['intent']:
            if "NONE" not in user_previous_action['cs']:
                ml_models.count(agent_action['ack_cs'], self.rec_I_agent)

    def vectorize(self):
        state_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if "I" in self.state["user_social_type"]:
            state_vector[0] = 1
        if "crew" in self.state["slots_requested"]:
            state_vector[1] = 1
        if "cast" in self.state["slots_requested"]:
            state_vector[2] = 1
        if "genre" in self.state["slots_requested"]:
            state_vector[3] = 1
        if "SD" in self.state["user_current_cs"]:
            state_vector[4] = 1
        if "PR" in self.state["user_current_cs"]:
            state_vector[5] = 1
        if "HE" in self.state["user_current_cs"]:
            state_vector[6] = 1
        if "VSN" in self.state["user_current_cs"]:
            state_vector[7] = 1
        if "yes" in self.state["user_action"]:
            state_vector[8] = 1
        if "no" in self.state["user_action"]:
            state_vector[9] = 1
        if "last_movie" in self.state["user_action"]:
            state_vector[10] = 1
        if "cast" in self.state["user_action"]:
            state_vector[11] = 1
        if "crew" in self.state["user_action"]:
            state_vector[12] = 1
        if "genre" in self.state["user_action"]:
            state_vector[13] = 1
        if "SD" in self.state["previous_agent_cs"]:
            state_vector[14] = 1
        if "PR" in self.state["previous_agent_cs"]:
            state_vector[15] = 1
        if "HE" in self.state["previous_agent_cs"]:
            state_vector[16] = 1
        if "VSN" in self.state["previous_agent_cs"]:
            state_vector[17] = 1
        if "greeting" in self.state["previous_agent_action"]:
            state_vector[18] = 1
        if "last_movie" in self.state["previous_agent_action"]:
            state_vector[19] = 1
        if "cast" in self.state["previous_agent_action"]:
            state_vector[20] = 1
        if "crew" in self.state["previous_agent_action"]:
            state_vector[21] = 1
        if "genre" in self.state["previous_agent_action"]:
            state_vector[22] = 1
        if "(movie)" in self.state["previous_agent_action"]:
            state_vector[23] = 1
        if "goodbye" in self.state["previous_agent_action"]:
            state_vector[24] = 1
        state_vector = np.asarray(state_vector)
        state_vector = state_vector[np.newaxis, :]
        return state_vector