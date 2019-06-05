import random
import numpy
import config
import json


class UserSimulator():
    def __init__(self):
        # User preferences
        self.pref_actor = []
        self.pref_director = []
        self.pref_genre = []

        # User type (P-Type or I-Type)
        self.user_type = ""

        # Number of recommendations wanted by the user (from 1 to 6 included)
        self.number_recos = 0
        self.current_number_recos = 0

        self.list_actions = []

        self.generate_user()
        self.load_actions_lexicon(config.USER_ACTIONS)

    def generate_user(self):
        self.set_preferences()
        self.set_type()
        self.set_number_recos()

    def set_number_recos(self):
        self.number_recos = numpy.random.choice(numpy.arange(1, 7), p=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15])

    def set_type(self):
        # The prior probabilities for a user being I-Type
        # or P-Type were learned from the Davos data using MLE.
        type = numpy.random.choice(numpy.arange(0, 2), p=[0.641, 0.359])
        if type == 0:
            self.user_type = "P"
        else:
            self.user_type = "I"

    def set_preferences(self):
        list_actors = []
        list_genres = []
        list_directors = []
        with open(config.USER_LIST_ACTORS) as f:
            for line in f:
                line_input = line.split("-")
                list_actors.append(line_input[0])
        with open(config.USER_LIST_GENRES) as f:
            for line in f:
                line_input = line.replace('\n', '')
                line_input = line_input.split("-")
                list_genres.append(line_input[0])
        with open(config.USER_LIST_DIRECTORS) as f:
            for line in f:
                line_input = line.split("-")
                list_directors.append(line_input[0])
        self.pref_actor = random.choice(list_actors)
        self.pref_director = random.choice(list_directors)
        self.pref_genre = random.choice(list_genres)

    def load_actions_lexicon(self, path):
        with open(path) as f:
            for line in f:
                line_input = line.replace('\n', '')
                self.list_actions.append(line_input)

    def next(self, agent_action):
        user_entity = ''
        entity_type = ''
        polarity = ''
        user_intention = "yes"

        # Todo User can say No to a request
        # Todo User can request things after being recommended a movie
        # Todo Add NLG

        if self.number_recos > self.current_number_recos:
            if "start" in agent_action['intent']:
                user_intention = "greeting"
            elif "request" in agent_action['intent']:
                if "genre" in agent_action['intent']:
                    user_intention = 'inform'
                    user_entity = self.pref_genre
                    entity_type = 'genre'
                    polarity = "+"
                elif "director" in agent_action['intent']:
                    user_intention = 'inform'
                    user_entity = self.pref_director
                    entity_type = 'cast'
                    polarity = "+"
                elif "actor" in agent_action['intent']:
                    user_intention = 'inform'
                    user_entity = self.pref_actor
                    entity_type = 'cast'
                    polarity = "+"
            elif "inform" in agent_action['intent']:
                user_intention = "request"
                self.current_number_recos += 1
            else:
                user_intention = "yes"
        else:
            user_intention = "no"
        user_action = self.msg_to_json(user_intention, user_entity, entity_type, polarity)
        return user_action

    def msg_to_json(self, intent, entity, entity_type, polarity):
        frame = {'intent': intent, 'entity': entity, 'entity_type': entity_type, 'polarity': polarity}
        return frame
