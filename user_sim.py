import random
import numpy
import config

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


        self.generate_user()

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
                line_input = line.split("-")
                list_genres.append(line_input[0])
        with open(config.USER_LIST_DIRECTORS) as f:
            for line in f:
                line_input = line.split("-")
                list_directors.append(line_input[0])
        self.pref_actor = random.choice(list_actors)
        self.pref_director = random.choice(list_directors)
        self.pref_genre = random.choice(list_genres)

    def next(self, agent_action):
        if "greeting" in agent_action:
            user_action = "greeting"
