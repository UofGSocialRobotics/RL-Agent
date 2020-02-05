import ml_models


class DialogState():

    def __init__(self):
        self.full_dialog = {}
        self.dialog_done = False
        self.turns = 0
        self.accepted_recos = 0
        self.state = {}
        self.reward = 0
        #Todo check what is what
        self.rec_I_user = [0, 0, 0, 0, 0]
        self.rec_I_agent = [0, 0, 0, 0, 0]
        self.rec_P_agent = 0

    def set_initial_state(self, user):
        self.full_dialog = {}
        self.dialog_done = False
        self.turns = 0
        self.accepted_recos = 0
        self.state = {}
        self.reward = 0
        # Todo check what is what
        self.rec_I_user = [0, 0, 0, 0, 0]
        self.rec_I_agent = [0, 0, 0, 0, 0]
        self.rec_P_agent = 0

        self.state["user_social_type"] = user.user_type
        self.state["user_reco_type"] = user.user_reco_pref
        self.state["slots_requested"] = []
        self.state["user_current_cs"] = False
        self.state["user_action"] = ''
        #self.state["previous_user_action"] = ''
        self.state["previous_agent_action"] = ''

    def update_state(self, agent_action, user_action, agent_previous_action, user_previous_action, user_type):
        self.turns += 1.0
        #if "inform" in user_action['intent'] and user_action['entity_type'] not in self.state["slots_requested"]:
        #    self.state["slots_requested"].append(user_action['entity_type'])
        if agent_action['intent'] in ["request(crew)"] and "crew" not in self.state["slots_requested"]:
            self.state["slots_requested"].append("crew")
        if agent_action['intent'] in ["request(cast)"] and "cast" not in self.state["slots_requested"]:
            self.state["slots_requested"].append("cast")
        if agent_action['intent'] in ["request(genre)"] and "genre" not in self.state["slots_requested"]:
            self.state["slots_requested"].append("genre")
        if "last_movie" in agent_action['intent'] and "last_movie" not in self.state["slots_requested"]:
            self.state["slots_requested"].append("last_movie")
        self.state["slots_requested"].sort()
        if "inform(movie)" in agent_action['intent'] and "yes" in user_action['intent']:
            self.accepted_recos += 1
        if user_action['cs']:
            self.state["user_current_cs"] = user_action['cs']
        self.state["user_action"] = user_action['intent']
        self.state["previous_agent_action"] = agent_action['intent']
        if "bye" in agent_action['intent']:
            self.dialog_done = True
        self.append_data_from_simulation(agent_action, user_action, agent_previous_action, user_previous_action, user_type)

    def append_data_from_simulation(self, agent_action, user_action, agent_previous_action, user_previous_action, user_type):
        if "P" in user_type:
            if "NONE" in user_action['cs'] and "NONE" in agent_action['cs']:
                self.rec_P_agent += 1
        else:
            if "NONE" not in user_action['cs']:
                ml_models.count(agent_action['cs'], self.rec_I_user)
            if "start" not in agent_previous_action['intent']:
                if "NONE" not in user_previous_action['cs']:
                    ml_models.count(agent_action['ack_cs'], self.rec_I_agent)

    def compute_reward(self, state, agent_action, user_number_wanted_recos):
        task_reward = 0
        rapport_reward = 0
        #Todo do not say bye before user gets to his limit

        #####################       Task Reward     #########################
        self.reward += -1
        if "request" in agent_action['intent'] and agent_action['intent'].replace('request(', '').replace(')', '') in state["slots_requested"]:
            self.reward += -30
        if "last_movie" in agent_action['intent'] and "last_movie" in state["slots_requested"]:
            self.reward += -30
        if self.dialog_done:
            #print("user wanted " + str(user_number_wanted_recos) + " recos and accepted " + str(self.state['recos']))
            self.reward = self.reward + (self.accepted_recos/user_number_wanted_recos) * 100
            task_reward = self.reward



        #####################       Social Reward     #########################
            data = []
            data.extend(self.rec_I_user)
            data.extend(self.rec_I_agent)
            rapport = ml_models.estimate_rapport(data)
            rapport_reward = ml_models.get_rapport_reward(rapport, self.rec_P_agent / self.turns, self.state["user_social_type"])
            self.reward = self.reward + rapport_reward
            #print("Rapport :" + str(rapport))
            #print("Reward total: " + str(self.reward))
            #print("Reward from Rapport: " + str(rapport_reward) + " and from Task: " + str(task_reward))

        #print(self.reward)
        return self.reward, task_reward, rapport_reward