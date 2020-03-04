import random
import config
import utils


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
        self.movies_list = []

        self.user_action = None

        self.movie = {'title': "", 'year': "", 'plot': "", 'actors': [], 'genres': [], 'poster': ""}
        self.nodes = {}
        self.user_model = {"liked_cast": [], "disliked_cast": [], 'liked_crew': [], 'disliked_crew': [],
                           "liked_genres": [], 'disliked_genres': [], 'liked_movies': [], 'disliked_movies': []}
        self.load_model(config.DM_MODEL)

    def init_agent(self):
        self.currState = "start"
        # Do we store the users preferences in a user model?
        self.store_pref = True
        self.movies_list = []

        self.user_action = None

        self.movie = {'title': "", 'year': "", 'plot': "", 'actors': [], 'genres': [], 'poster': ""}
        self.nodes = {}
        self.user_model = {"liked_cast": [], "disliked_cast": [], 'liked_crew': [], 'disliked_crew': [],
                           "liked_genres": [], 'disliked_genres': [], 'liked_movies': [], 'disliked_movies': []}
        self.load_model(config.DM_MODEL)

    # Parse the model.csv file and transform that into a dict of Nodes representing the scenario
    def load_model(self, path):
        with open(path) as f:
            for line in f:
                line_input = line.replace('\n', '')
                line_input = line_input.split(",")
                node = DMNode(line_input[0], line_input[1], line_input[2])
                for i in range(3, len(line_input)):
                    if "-" in line_input[i]:
                        node.add(line_input[i])
                self.nodes[node.stateName] = node

    def next(self, state):
        self.user_action = state.state['current_user_action']

        if state.turns == 0:
            next_state = 'greeting()'
        else:
            next_state = self.nodes.get(self.currState).get_action(self.user_action['intent'])

            # Store entities (actors,directors, genres) in the user frame
            if self.store_pref and "inform" in self.user_action['intent']:
                if '+' in self.user_action['polarity']:
                    if 'cast' in self.user_action['entity_type'] or 'actor' in self.user_action['entity_type']:
                        self.user_model["liked_cast"].append(self.user_action['entity'])
                    elif 'genre' in self.user_action['entity_type']:
                        self.user_model["liked_genres"].append(self.user_action['entity'])
                elif '-' in self.user_action['polarity']:
                    if 'cast' in self.user_action['entity_type']:
                        self.user_model["disliked_cast"].append(self.user_action['entity'])
                    elif 'genre' in self.user_action['entity_type']:
                        self.user_model["disliked_genre"].append(self.user_action['entity'])

        if self.currState in ("inform(movie)", "inform(plot)", "inform(actor)", "inform(genre)"):
            if "affirm" in self.user_action['intent']:
                self.user_model['liked_movies'].append(self.movie['title'])
            elif any(s in self.user_action['intent'] for s in ('request(more)', 'inform(watched)', 'negate')):
                self.user_model['disliked_movies'].append(self.movie['title'])

        # Todo: Uncomment if real recommendations are needed.
        # Get a movie recommendation title
        # if "inform(movie)" in next_state:
        #     self.movie['title'] = self.recommend()
        #     movie_info = utils.get_movie_info(self.movie['title'])
        #     self.movie['plot'] = movie_info.get("Plot")
        #     self.movie['actors'] = movie_info.get("Actors")
        #     self.movie['genres'] = movie_info.get("Genre")

        self.currState = next_state
        agent_cs = self.pick_cs()
        ack_cs = self.pick_cs()
        new_msg = self.msg_to_json(next_state, self.movie, ack_cs, agent_cs)
        self.user_action = None

        return new_msg

    def msg_to_json(self, intention, movie, ack_cs, cs):
        intent, entity_type = utils.parse_intention(intention)
        frame = {'intent': intent, 'entity_type': entity_type, 'movie': movie, 'ack_cs': ack_cs, 'cs': cs}
        return frame

    def recommend(self):
        if not self.movies_list:
            self.movies_list = utils.query_blended_movies_list(self.user_model)
        for movie in self.movies_list:
            if movie['title'] not in self.user_model['liked_movies'] and movie['title'] not in self.user_model['disliked_movies']:
                return movie['title']


    def pick_cs(self):
        agent_cs = random.choice(config.CS_LABELS)
        return agent_cs

# A node corresponds to a specific state of the dialogue. It contains:
# - a state ID (int)
# - a state name (String)
# - a default state (String) which represents the next state by default, whatever the user says.
# - a set of rules (dict) mapping a specific user intent to another state (e.g. yes-inform() means that if the user says
#   yes, the next state will be inform())
class DMNode:
    def __init__(self, state_name, state_default, state_ack):
        self.stateName = state_name
        self.stateDefault = state_default
        if state_ack.lower() == "true":
            self.stateAck = True
        else:
            self.stateAck = False
        self.rules = {}

    def add(self, string):
        intents = string.split("-")
        self.rules[intents[0]] = intents[1]

    def get_action(self, user_intent):
        if user_intent in self.rules:
            return self.rules.get(user_intent)
        else:
            return self.stateDefault
