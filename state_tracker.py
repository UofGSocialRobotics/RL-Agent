import ml_models


class DialogState():

    def __init__(self):
        self.full_dialog = {}
        self.dialog_done = False
        self.turns = 0
        self.state = {}
        self.reward = 0
        #Todo check what is what
        self.rec_I_user = [0, 0, 0, 0, 0]
        self.rec_I_agent = [0, 0, 0, 0, 0]
        self.rec_P_agent = 0

    def set_initial_state(self, user):
        self.state["user_social_type"] = user.user_type
        self.state["user_reco_type"] = user.user_reco_pref
        self.state["slots_requested"] = []
        self.state["recos"] = 0
        self.state["user_current_cs"] = False
        self.state["user_action"] = ''
        self.state["previous_user_action"] = ''
        self.state["previous_agent_action"] = ''

    def update_state(self, agent_action, user_action, agent_previous_action, user_previous_action, user_type):
        self.turns += 1.0
        #if "inform" in user_action['intent'] and user_action['entity_type'] not in self.state["slots_requested"]:
        #    self.state["slots_requested"].append(user_action['entity_type'])
        if agent_action in ["request(crew)"] and "crew" not in self.state["slots_requested"]:
            self.state["slots_requested"].append("crew")
        if agent_action in ["request(cast)"] and "cast" not in self.state["slots_requested"]:
            self.state["slots_requested"].append("cast")
        if agent_action in ["request(genre)"] and "genre" not in self.state["slots_requested"]:
            self.state["slots_requested"].append("genre")
        if "last_movie" in agent_action['intent']:
            self.state["slots_requested"].append("last_movie")
        if "inform(movie)" in agent_action['intent'] and "yes" in user_action['intent']:
            self.state['recos'] += 1
        if user_action['cs']:
            self.state["user_current_cs"] = user_action['cs']
        self.state["user_action"] = user_action['intent']
        self.state["agent_previous_action"] = agent_action['intent']
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

    # def compute_simple_reward(self):
    #     self.reward += -1
    #     if self.dialog_done:
    #         if "genre" in self.state['slots_requested']:
    #             if "Nov" in self.state["user_reco_type"]:
    #                 self.reward += -30.0
    #             else:
    #                 self.reward += 30.0
    #         else:
    #             if "Nov" in self.state["user_reco_type"]:
    #                 self.reward += 30.0
    #             else:
    #                 self.reward += -30.0
    #         if self.turns < 3:
    #             if "P" in self.state["user_social_type"]:
    #                 self.reward += 30.0
    #             else:
    #                 self.reward += -30.0
    #         else:
    #             if "P" in self.state["user_social_type"]:
    #                 self.reward += -30.0
    #             else:
    #                 self.reward += 30.0
    #     return self.reward

    def compute_reward(self, state, agent_action):
        task_reward = 0
        #Todo do not say bye before user gets to his limit

        #####################       Task Reward     #########################
        self.reward += -1
        if "request" in agent_action['intent'] and agent_action['intent'].replace('request(', '').replace(')', '') in state["slots_requested"]:
            self.reward += -30
        if "last_movie" in agent_action['intent'] and "last_movie" in state["slots_requested"]:
            self.reward += -30
        if self.dialog_done:
            if self.state['recos'] == 0:
                self.reward += -50
            self.reward = self.reward + self.state['recos'] * 100
            task_reward = self.reward
            #self.reward = self.reward + (len(self.state["slots_requested"]) * 20)



        #####################       Social Reward     #########################
            data = []
            data.extend(self.rec_I_user)
            data.extend(self.rec_I_agent)
            rapport = ml_models.estimate_rapport(data)
            rapport_reward = ml_models.get_rapport_reward(rapport, self.rec_P_agent / self.turns, self.state["user_social_type"])
            self.reward = self.reward + rapport_reward
            #print("Rapport :" + str(rapport))
            print("Reward total: " + str(self.reward))
            print("Reward from Rapport: " + str(rapport_reward) + " and from Task: " + str(task_reward))

        #print(self.reward)
        return self.reward