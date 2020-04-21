import random
import pandas
import numpy as np
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

        self.actions = self.load_actions_lexicon(config.AGENT_ACTION_SPACE)

        self.movie = {'title': "", 'year': "", 'plot': "", 'actors': [], 'genres': [], 'poster': ""}
        self.nodes = {}
        self.user_model = {"liked_cast": [], "disliked_cast": [], 'liked_crew': [], 'disliked_crew': [],
                           "liked_genres": [], 'disliked_genres': [], 'liked_movies': [], 'disliked_movies': []}
        self.load_model(config.DM_MODEL)


        # This is for the RL_agent
        self.qtable_columns, self.actions = self.load_actions_lexicon(config.AGENT_ACTION_SPACE)

        self.qtable = pandas.DataFrame(0, index=[], columns=self.qtable_columns)

        self.learning_rate = config.LEARNING_RATE
        self.gamma = config.GAMMA
        self.exploration_rate = config.EPSILON
        self.exploration_min = config.EPSILON_MIN
        self.exploration_decay = config.EPSILON_DECAY





    ####################################################################################
    ##########################     RULE-BASED FUNCTIONS    #############################
    ####################################################################################


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

        if state.accepted_recos >= 1:
            next_state = "bye()"

        ack_cs, agent_cs = self.pick_cs(next_state)
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


    def pick_cs(self, action):
        #Todo check valid CS
        action = action.split('(')
        action_pool = []
        for actions in self.actions:
            if action[0] in actions:
                action_pool.append(actions)
        agent_action = random.choice(action_pool)
        ack_cs = agent_action[0]
        agent_cs = agent_action[2]
        return ack_cs, agent_cs





    ####################################################################################
    ##########################     RL FUNCTIONS    #############################
    ####################################################################################

    def load_actions_lexicon(self, path):
        with open(path) as f:
            list_actions = []
            list_columns = []
            for line in f:
                line_qtable_columns = line.replace('\n','')
                line_input = line.replace('\n','').split(',')
                list_columns.append(line_qtable_columns)
                list_actions.append(line_input)
        return list_columns, list_actions


    def update_qtables(self, prev_state, current_state, agent_action, reward):
        action = agent_action['ack_cs'] + "," + agent_action['intent'] + "," + agent_action['cs']
        if str(prev_state) in self.qtable.index:
            if str(current_state) not in self.qtable.index:
                self.qtable.loc[str(current_state)] = 0.0
            discounted_reward = (1 - self.learning_rate) * self.qtable.at[str(prev_state), action] + self.learning_rate * (
                        reward + self.gamma * np.max(self.qtable.loc[str(current_state), :]))
            self.qtable.at[str(prev_state), action] = discounted_reward
        else:
            self.qtable.loc[str(prev_state)] = 0.0
            if str(current_state) not in self.qtable.index:
                self.qtable.loc[str(current_state)] = 0.0
            discounted_reward = (1 - self.learning_rate) * self.qtable.at[str(prev_state), action] + self.learning_rate * (reward + self.gamma * np.max(self.qtable.loc[str(current_state), :]))
            self.qtable.at[str(prev_state), action] = discounted_reward


    def next_rl(self, state):
        if random.uniform(0, 1) > config.EPSILON:
            agent_action = self.next_best(state)
        else:
            agent_action = random.choice(self.actions)
        entity_type = self.pick_slot(state, agent_action)
        new_msg = self.msg_to_json2(agent_action, entity_type)
        return new_msg

    def next_best(self, state):
        current_state = self.qtable.loc[str(state)]
        action = current_state.idxmax()
        return action

    def pick_slot(self, state, agent_action):
        if 'request' in agent_action[1]:
            if "genre" not in state.state["slots_filled"]:
                return "genre"
            if "actor" not in state.state["slots_filled"]:
                return "actor"
            if "director" not in state.state["slots_filled"]:
                return "director"
            else:
                return random.choice(["genre","actor","director"])
        elif 'introduce' in agent_action[1]:
            if "role" not in state.state["introduce_slots"]:
                return "role"
            if "last_movie" not in state.state["introduce_slots"]:
                return "last_movie"
            if "reason_like" not in state.state["introduce_slots"]:
                return "reason_like"
            else:
                return random.choice(["role","last_movie","reason_like"])
        elif 'inform' in agent_action[1]:
            return "movie"
        else:
            return ""

    def msg_to_json2(self, intention, entity_type):
        frame = {'intent': intention[1], 'entity_type': entity_type, 'ack_cs': intention[0], 'cs': intention[2]}
        return frame








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
