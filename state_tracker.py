class DialogState():

    def __init__(self):
        self.state = {}
        self.reward = 0


    def set_initial_state(self, user):
        self.state["user_social_type"] = user.user_type
        self.state["user_reco_type"] = user.user_reco_pref
        self.state["slots_requested"] = []
        self.state["recos"] = 0
        self.state["user_current_cs"] = False
        self.state["user_action"] = ''
        self.state["previous_user_action"] = ''
        self.state["previous_agent_action"] = ''
        self.turns = 0

        self.dialog_done = False
        self.full_dialog = {}

        # check what is what
        rec_I_user = [0, 0, 0, 0, 0]
        rec_I_agent = [0, 0, 0, 0, 0]
        rec_P_agent = 0

        self.user_model = {"liked_cast": [], "disliked_cast": [], 'liked_crew': [], 'disliked_crew': [],
                           "liked_genres": [], 'disliked_genres': [], 'liked_movies': [], 'disliked_movies': []}


    def update_state(self, agent_action, user_action):
        self.turns += 1.0
        if "inform" in user_action['intent'] and user_action['entity_type'] not in self.state["slots_requested"]:
            self.state["slots_requested"].append(user_action['entity_type'])
        if "inform(movie)" in agent_action['intent'] and "yes" in user_action['intent']:
            self.state['recos'] += 1
        if user_action['cs']:
            self.state["user_current_cs"] = user_action['cs']
        self.state["user_action"] = user_action['intent']
        self.state["agent_previous_action"] = agent_action['intent']
        if "bye" in agent_action['intent']:
            self.dialog_done = True

    def compute_simple_reward(self):
        self.reward += -1
        if self.dialog_done:
            if "genre" in self.state['slots_requested']:
                if "Nov" in self.state["user_reco_type"]:
                    self.reward += -30.0
                else:
                    self.reward += 30.0
            else:
                if "Nov" in self.state["user_reco_type"]:
                    self.reward += 30.0
                else:
                    self.reward += -30.0
            if self.turns < 3:
                if "P" in self.state["user_social_type"]:
                    self.reward += 30.0
                else:
                    self.reward += -30.0
            else:
                if "P" in self.state["user_social_type"]:
                    self.reward += -30.0
                else:
                    self.reward += 30.0
        return self.reward

    def compute_reward(self):
        self.reward += -1
        if self.dialog_done:
            self.reward = self.state['recos'] * 50
        return self.reward




        #rec_I_user, rec_I_agent, rec_P_agent = re.append_data_from_simulation(user_action, agent_action, agent_previous_action, rec_I_user, rec_I_agent, rec_P_agent, user.user_type)

        #agent_previous_action = agent_action

    #data.extend(rec_I_user)
    #data.extend(rec_I_agent)
    #re.estimate_rapport(data)
    #reward = re.get_rapport_reward(re.estimate_rapport(data), rec_P_agent/turns, user.user_type)
    #print("Rapport :" + str(re.estimate_rapport(data)))
    #print("Reward :" + str(reward))
    #print(agent.cs_qtable)