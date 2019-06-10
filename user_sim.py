import random
import numpy
import config
import utils


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

        self.movie_agenda = []

        if config.GENERATE_SENTENCE:
            self.sentenceDB = utils.load_user_sentence_model(config.USER_SENTENCES)

        if config.GENERATE_VOICE:
            self.engine = utils.set_voice_engine("U", config.USER_VOICE)

        self.list_actions = []

        self.generate_user()
        self.load_actions_lexicon(config.USER_ACTIONS)

    def generate_user(self):
        self.set_preferences()
        self.set_type()
        self.set_number_recos()

    def set_number_recos(self):
        self.number_recos = numpy.random.choice(numpy.arange(1, 7), p=config.PROBA_NUMBER_MOVIES)

    def set_type(self):
        # The prior probabilities for a user being I-Type
        # or P-Type were learned from the Davos data using MLE.
        self.user_type = numpy.random.choice(config.ITEMS_USER_TYPE, p=config.PROBA_USER_TYPE)

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
        self.pref_actor = numpy.random.choice([[], random.choice(list_actors)], p=config.PROBA_NO_ACTOR)
        self.pref_director = numpy.random.choice([[], random.choice(list_directors)], p=config.PROBA_NO_DIRECTOR)
        self.pref_genre = numpy.random.choice([[], random.choice(list_genres)], p=config.PROBA_NO_GENRE)

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

        # Todo Add Acks
        # Todo Add CS

        if self.number_recos > self.current_number_recos:
            if "start" in agent_action['intent']:
                user_intention = "greeting"
            elif "request" in agent_action['intent']:
                if "genre" in agent_action['intent']:
                    if self.pref_genre:
                        user_intention = 'inform(genre)'
                        user_entity = self.pref_genre
                        entity_type = 'genre'
                        polarity = "+"
                    else:
                        user_intention = 'no'
                elif "director" in agent_action['intent']:
                    if self.pref_director:
                        user_intention = 'inform(director)'
                        user_entity = self.pref_director
                        entity_type = 'cast'
                        polarity = "+"
                    else:
                        user_intention = 'no'
                elif "actor" in agent_action['intent']:
                    if self.pref_actor:
                        user_intention = 'inform(actor)'
                        user_entity = self.pref_actor
                        entity_type = 'cast'
                        polarity = "+"
                    else:
                        user_intention = 'no'
                elif "why" in agent_action['intent']:
                    user_intention = 'inform(why)'
                elif "opinion" in agent_action['intent']:
                    user_intention = 'inform(opinion)'
            elif "inform" in agent_action['intent']:
                user_intention = numpy.random.choice(config.ITEMS_REQUEST_AFTER_MOVIE, p=config.PROBA_REQUEST_AFTER_MOVIE)
                self.current_number_recos += 1
            else:
                user_intention = "yes"
        else:
            # Todo add condition so that the user can inform(why) even after the last movie.
            # Todo User will say no to request(why) otherwise
            if "why" in agent_action['intent']:
                user_intention = 'inform(why)'
            elif "opinion" in agent_action['intent']:
                user_intention = 'inform(opinion)'
            else:
                user_intention = "no"

        # Todo build Social Reasoner
        #user_cs = 'HE'
        user_cs = random.choice(config.CS_LABELS)
        user_action = self.msg_to_json(user_intention, user_cs, user_entity, entity_type, polarity)
        return user_action

    def msg_to_json(self, intent, cs, entity, entity_type, polarity):
        frame = {'intent': intent, 'cs': cs, 'entity': entity, 'entity_type': entity_type, 'polarity': polarity}
        return frame