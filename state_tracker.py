class DialogState():

    def __init__(self):
        self.state = {}


    def set_initial_state(self, user):
        self.state["user_social_type"] = user.user_type
        self.state["user_reco_type"] = user.user_reco_pref
        self.state["slots_requested"] = []
        self.state["recos"] = 0
        self.state["user_current_cs"] = False
        self.state["user_action"] = {}
        self.state["previous_user_action"] = {}
        self.state["turns"] = 0

        self.dialog_done = False
        self.full_dialog = {}

        # check what is what
        rec_I_user = [0, 0, 0, 0, 0]
        rec_I_agent = [0, 0, 0, 0, 0]
        rec_P_agent = 0

    def update_state(self, agent_action, user_action):
        self.state["turns"] += 1
        if "inform" in user_action['intent']:
            self.state["slots_requested"].append(user_action['entity_type'])
