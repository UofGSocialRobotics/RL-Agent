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
        # User type (Nov-type or Sim-Type)
        self.user_reco_pref = ""

        # Number of recommendations wanted by the user (from 1 to 6 included)
        self.number_recos = 0
        self.accepted_recos = 0
        self.current_number_recos = 0

        self.movie_agenda = list(config.ITEMS_REQUEST_AFTER_MOVIE)
        self.movie_agenda_probas = list(config.PROBA_REQUEST_AFTER_MOVIE)

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
        self.set_other_features()

    def set_other_features(self):
        self.accepted_recos = 0
        self.current_number_recos = 0

        self.movie_agenda = list(config.ITEMS_REQUEST_AFTER_MOVIE)
        self.movie_agenda_probas = list(config.PROBA_REQUEST_AFTER_MOVIE)

        #self.list_actions = []

        #self.load_actions_lexicon(config.USER_ACTIONS)

    def set_number_recos(self):
        self.number_recos = numpy.random.choice(numpy.arange(1, 7), p=config.PROBA_NUMBER_MOVIES)

    def set_type(self):
        # The prior probabilities for a user being I-Type
        # or P-Type were learned from the Davos data using MLE.
        self.user_type = numpy.random.choice(config.ITEMS_USER_TYPE, p=config.PROBA_USER_TYPE)
        self.user_reco_pref = numpy.random.choice(config.ITEMS_USER_RECO_PREF, p=config.PROBA_USER_RECO_PREF)

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

    def next(self, agent_action, complete_state):
        user_entity = ''
        entity_type = ''
        polarity = ''
        user_intention = "yes"
        state = complete_state.state

        if self.number_recos > self.accepted_recos and complete_state.turns < config.MAX_STEPS:
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
                elif "crew" in agent_action['intent']:
                    if self.pref_director:
                        user_intention = 'inform(director)'
                        user_entity = self.pref_director
                        entity_type = 'crew'
                        polarity = "+"
                    else:
                        user_intention = 'no'
                elif "cast" in agent_action['intent']:
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
            # Todo Do something better here
            # Todo Check when to reinit the movie_agenda
            elif "inform" in agent_action['intent']:
                user_intention = self.response_to_inform_movie(agent_action, config.INTERACTION_MODE, state)
            else:
                user_intention = "yes"
        else:
            if "why" in agent_action['intent']:
                user_intention = 'inform(why)'
            elif "opinion" in agent_action['intent']:
                user_intention = 'inform(opinion)'
            else:
                user_intention = "bye"

        user_cs = self.pick_cs()

        user_action = self.msg_to_json(user_intention, user_cs, user_entity, entity_type, polarity)
        return user_action

    def pick_cs(self):
        # Todo build Social Reasoner
        if "P" in self.user_type:
            user_cs = 'NONE'
        else:
            user_cs = random.choice(config.CS_LABELS)
        return user_cs

    def response_to_inform_movie(self, agent_action, mode, state):
        if "RL" in mode:
            if "Nov" in state["user_reco_type"]:
                if "genre" in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.2, 0.8]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.25, 0.75]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.25, 0.75]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.25, 0.75]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.7, 0.3]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.4, 0.6]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.4, 0.6]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.85, 0.15]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
            else:
                if "genre" in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.85, 0.15]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.7, 0.3]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.7, 0.3]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.7, 0.3]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.3, 0.7]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.3, 0.7]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.3, 0.7]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
                elif "genre" not in state["slots_requested"] and "cast" not in state["slots_requested"] and "crew" not in state["slots_requested"]:
                    self.movie_agenda = ["yes", "no"]
                    self.movie_agenda_probas = [0.15, 0.85]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                    if "yes" in user_intention:
                        self.accepted_recos += 1
                    self.current_number_recos += 1
        else:
            if "(movie)" in agent_action['intent']:
                self.movie_agenda = list(config.ITEMS_REQUEST_AFTER_MOVIE)
                self.movie_agenda_probas = list(config.PROBA_REQUEST_AFTER_MOVIE)
                user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                self.current_number_recos += 1
            elif "(genre)" in agent_action['intent']:
                if 'request(actor)' and 'request(plot)' in self.movie_agenda:
                    self.movie_agenda.remove('request(genre)')
                    self.movie_agenda_probas = [0.2, 0.15, 0.1, 0.2, 0.2, 0.15]
                    if 'inform(watched)' in self.movie_agenda:
                        self.movie_agenda.remove('inform(watched)')
                        self.movie_agenda_probas = [0.25, 0.15, 0.15, 0.15, 0.3]
                    print(self.movie_agenda)
                    print(self.movie_agenda_probas)
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                else:
                    self.movie_agenda = ['yes', 'no', 'request(another)']
                    self.movie_agenda_probas = [0.33, 0.33, 0.34]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
            elif "(actor)" in agent_action['intent']:
                if 'request(genre)' and 'request(plot)' in self.movie_agenda:
                    self.movie_agenda.remove('request(actor)')
                    self.movie_agenda_probas = [0.2, 0.15, 0.1, 0.2, 0.2, 0.15]
                    if 'inform(watched)' in self.movie_agenda:
                        self.movie_agenda.remove('inform(watched)')
                        self.movie_agenda_probas = [0.25, 0.15, 0.15, 0.15, 0.3]
                    print(self.movie_agenda)
                    print(self.movie_agenda_probas)
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                else:
                    self.movie_agenda = ['yes', 'no', 'request(another)']
                    self.movie_agenda_probas = [0.33, 0.33, 0.34]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
            elif "(plot)" in agent_action['intent']:
                if 'request(actor)' and 'request(genre)' in self.movie_agenda:
                    self.movie_agenda.remove('request(plot)')
                    self.movie_agenda_probas = [0.2, 0.15, 0.1, 0.15, 0.25, 0.15]
                    if 'inform(watched)' in self.movie_agenda:
                        self.movie_agenda.remove('inform(watched)')
                        self.movie_agenda_probas = [0.25, 0.15, 0.15, 0.15, 0.3]
                    print(self.movie_agenda)
                    print(self.movie_agenda_probas)
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)
                else:
                    self.movie_agenda = ['yes', 'no', 'request(another)']
                    self.movie_agenda_probas = [0.33, 0.33, 0.34]
                    user_intention = numpy.random.choice(self.movie_agenda, p=self.movie_agenda_probas)

        return user_intention

    def msg_to_json(self, intent, cs, entity, entity_type, polarity):
        frame = {'intent': intent, 'cs': cs, 'entity': entity, 'entity_type': entity_type, 'polarity': polarity}
        return frame
