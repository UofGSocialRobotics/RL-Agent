import ml_models
import utils
import config


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
        self.reward = 0

        self.rec_I_user = [0, 0, 0, 0, 0]
        self.rec_I_agent = [0, 0, 0, 0, 0]
        self.rec_P_agent = 0

        self.state["introduce_slots"] = []
        self.state["slots_filled"] = []
        self.state["previous_user_action"] = config.USER_ACTION
        self.state["current_user_action"] = config.USER_ACTION
        self.state["previous_agent_action"] = config.AGENT_ACTION
        self.state["current_agent_action"] = config.AGENT_ACTION

    def update_state(self, agent_action, user_action, agent_previous_action, user_previous_action):
        self.turns += 1.0

        #update slots
        if "request" in agent_action['intent']:
            if "director" in agent_action['entity_type'] and "director" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("director")
            if "actor" in agent_action['entity_type'] and "actor" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("actor")
            if "genre" in agent_action['entity_type'] and "genre" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("genre")
        if "introduce" in agent_action['intent']:
            if "role" in agent_action['entity_type'] and "role" not in self.state["introduce_slots"]:
                self.state["introduce_slots"].append("role")
            if "last_movie" in agent_action['entity_type'] and "last_movie" not in self.state["introduce_slots"]:
                self.state["introduce_slots"].append("last_movie")
            if "reason_like" in agent_action['entity_type'] and "reason_like" not in self.state["introduce_slots"]:
                self.state["introduce_slots"].append("reason_like")
        if "inform" in user_action['intent']:
            if "director" in user_action['entity_type'] and "director" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("director")
            if "actor" in user_action['entity_type'] and "actor" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("actor")
            if "genre" in user_action['entity_type'] and "genre" not in self.state["slots_filled"]:
                self.state["slots_filled"].append("genre")
        self.state["slots_filled"].sort()
        self.state["introduce_slots"].sort()

        #update agent and user actions
        self.state["current_user_action"] = user_action
        self.state["previous_user_action"] = user_previous_action
        self.state["current_agent_action"] = agent_action
        self.state["previous_agent_action"] = agent_previous_action


        #update recos
        if "inform" in agent_action['intent'] and "movie" in agent_action['entity_type']:
            self.delivered_recos += 1
        if "inform" in agent_action['intent']:
            if "affirm" in user_action['intent']:
                self.accepted_recos += 1

        if "bye" in user_action['intent']:
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

    def encode_state(self, agent_action_encoder, user_action_encoder):
        # Encoding agent actions
        agent_action = utils.transform_agent_action(self.state["current_agent_action"])
        agent_action = agent_action_encoder.transform(agent_action)
        previous_agent_action = utils.transform_agent_action(self.state["previous_agent_action"])
        previous_agent_action = agent_action_encoder.transform(previous_agent_action)
        # Encoding User actions
        user_action = utils.transform_user_action(self.state["current_user_action"])
        user_action = user_action_encoder.transform(user_action)
        previous_user_action = utils.transform_user_action(self.state["previous_user_action"])
        previous_user_action = user_action_encoder.transform(previous_user_action)
        # Todo Encode Slots
        slots = self.encode_slots()
        recos_good = self.encode_recos(self.accepted_recos)
        recos_done = self.encode_recos(self.delivered_recos)

        # Encoding State
        state = user_action.toarray().tolist()[0] + slots + recos_good + recos_done
        #state = previous_agent_action.toarray().tolist()[0] + previous_user_action.toarray().tolist()[0] + agent_action.toarray().tolist()[0] + user_action.toarray().tolist()[0]
        return agent_action, state

    def get_state(self):
        # Encoding agent actions
        agent_action = utils.transform_agent_action(self.state["current_agent_action"])
        previous_agent_action = utils.transform_agent_action(self.state["previous_agent_action"])
        # Encoding User actions
        user_action = utils.transform_user_action(self.state["current_user_action"])
        # Todo Encode Slots
        slots = self.encode_slots()
        recos_good = self.encode_recos(self.accepted_recos)
        recos_done = self.encode_recos(self.delivered_recos)

        # Encoding State
        state = str(agent_action[1]) + str(user_action) + str(slots) + str(recos_good) #+ str(recos_done)
        #state = previous_agent_action.toarray().tolist()[0] + previous_user_action.toarray().tolist()[0] + agent_action.toarray().tolist()[0] + user_action.toarray().tolist()[0]
        return state

    def encode_slots(self):
        mandatory = [0,0,0]
        optional = [0,0,0]
        if "actor" in self.state["slots_filled"]:
            mandatory[0] = 1
        if "director" in self.state["slots_filled"]:
            mandatory[1] = 1
        if "genre" in self.state["slots_filled"]:
            mandatory[2] = 1
        if "role" in self.state["introduce_slots"]:
            optional[0] = 1
        if "last_movie" in self.state["introduce_slots"]:
            optional[1] = 1
        if "reason_like" in self.state["introduce_slots"]:
            optional[2] = 1
        return (mandatory + optional)

    def encode_recos(self,number):
        vector = [0,0,0]
        if number == 1:
            vector = [0, 0, 1]
        if number == 2:
            vector = [0, 1, 0]
        if number == 3:
            vector = [0, 1, 1]
        if number == 4:
            vector = [1, 0, 0]
        if number == 5:
            vector = [1, 0, 1]
        if number == 6:
            vector = [1, 1, 0]
        if number == 7:
            vector = [1, 1, 1]
        return (vector)
