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

        self.currState = "start"
        # Do we store the users preferences in a user model?
        self.store_pref = True

        self.cs_qtable = pandas.DataFrame(0, index=config.CS_LABELS, columns=config.CS_LABELS)
        print(self.cs_qtable)

        self.user_action = None
        self.movies_list = []

        self.movie = {'title': "", 'year': "", 'plot': "", 'actors': [], 'genres': [], 'poster': ""}
        self.actions = []
        self.user_model = {"liked_cast": [], "disliked_cast": [], 'liked_crew': [], 'disliked_crew': [],
                           "liked_genres": [], 'disliked_genres': [], 'liked_movies': [], 'disliked_movies': []}
        self.load_actions(config.AGENT_ACTIONS)

    # Parse the model.csv file and transform that into a dict of Nodes representing the scenario
    def load_actions(self, path):
        with open(path) as f:
            for line in f:
                if "#" not in line:
                    self.actions.append(line.replace("\n",""))

    def next(self, state):
        if state["turns"] == 0:
            next_state = "greeting"
        else:
            next_state = random.choice(self.actions)

        self.actions.remove(next_state)
        #print(str(len(self.actions)) + "     " + str(self.actions))
        agent_cs = self.pick_cs()
        ack_cs = self.pick_cs()
        new_msg = self.msg_to_json(next_state, self.movie, ack_cs, agent_cs)

        return new_msg

    def msg_to_json(self, intention, movie, ack_cs, cs):
        frame = {'intent': intention, 'movie': movie, 'ack_cs': ack_cs, 'cs': cs}
        return frame

    def recommend(self):
        if not self.movies_list:
            self.movies_list = utils.query_blended_movies_list(self.user_model)
        for movie in self.movies_list:
            if movie['title'] not in self.user_model['liked_movies'] and movie['title'] not in self.user_model[
                'disliked_movies']:
                return movie['title']

    def pick_cs(self):
        agent_cs = random.choice(config.CS_LABELS)
        return agent_cs
