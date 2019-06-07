import random
import urllib.request
import json
from pathlib import Path
import config
import utils


class Agent():
    def __init__(self):

        if config.GENERATE_SENTENCE:
            self.sentenceDB = utils.load_agent_sentence_model(config.AGENT_SENTENCES)

        if config.GENERATE_VOICE:
            self.engine = utils.set_voice_engine(config.AGENT_VOICE)

        self.currState = "start"
        # Do we store the users preferences in a user model?
        self.store_pref = True

        self.user_action = None

        self.movie = {'title': "", 'year': "", 'plot': "", 'actors': [], 'genres': [], 'poster': ""}
        self.nodes = {}
        self.user_model = {"liked_cast": [], "disliked_cast": [], "liked_genres": [], 'disliked_genres': [],
                           'liked_movies': [], 'disliked_movies': []}
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

    def next(self, msg):
        self.user_action = msg
        # Store entities (actors,directors, genres) in the user frame
        if self.store_pref and "inform" in self.user_action['intent']:
            if '+' in self.user_action['polarity']:
                if 'cast' in self.user_action['entity_type']:
                    self.user_model["liked_cast"].append(self.user_action['entity'])
                elif 'genre' in self.user_action['entity_type']:
                    self.user_model["liked_genres"].append(self.user_action['entity'])
            elif '-' in self.user_action['polarity']:
                if 'cast' in self.user_action['entity_type']:
                    self.user_model["disliked_cast"].append(self.user_action['entity'])
                elif 'genre' in self.user_action['entity_type']:
                    self.user_model["disliked_genre"].append(self.user_action['entity'])

        next_state = self.nodes.get(self.currState).get_action(self.user_action['intent'])

        if self.currState in ("inform(movie)", "inform(plot)", "inform(actor)", "inform(genre)"):
            if "yes" in self.user_action['intent']:
                self.user_model['liked_movies'].append(self.movie['title'])
            elif any(s in self.user_action['intent'] for s in ('request', 'no')):
                self.user_model['disliked_movies'].append(self.movie['title'])

        # Get a movie recommendation title
        if "inform(movie)" in next_state:
            self.movie['title'] = self.recommend()
            self.set_movie_info(self.movie['title'])

        self.currState = next_state
        new_msg = self.msg_to_json(next_state, self.movie)
        self.user_action = None

        return new_msg

    def msg_to_json(self, intention, movie):
        frame = {'intent': intention, 'movie': movie}
        return frame

    def recommend(self):
        movies_list = self.queryMoviesList()
        for movie in movies_list:
            if movie['title'] not in self.user_model['liked_movies'] and movie['title'] not in self.user_model['disliked_movies']:
                return movie['title']

    def queryMoviesList(self):
        movies_with_cast_list = []
        movies_with_genres_list = []
        if not self.user_model['liked_genres'] and not self.user_model['liked_cast']:
            query_url = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + config.MOVIE_DB_PROPERTY
            data = urllib.request.urlopen(query_url)
            result = data.read()
            movies = json.loads(result)
            return movies['results']
        if self.user_model['liked_genres']:
            genre_id = self.get_genre_id(self.user_model['liked_genres'][-1].lower())
            query_url = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(
                genre_id) + config.MOVIE_DB_PROPERTY
            data = urllib.request.urlopen(query_url)
            result = data.read()
            movies = json.loads(result)
            movies_with_genres_list = movies['results']
        if self.user_model['liked_cast']:
            cast_id = self.get_cast_id(self.user_model['liked_cast'][-1].lower())
            query_url = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_people=" + str(
                cast_id) + config.MOVIE_DB_PROPERTY
            data = urllib.request.urlopen(query_url)
            result = data.read()
            movies = json.loads(result)
            movies_with_cast_list = movies['results']
        if movies_with_genres_list:
            if movies_with_cast_list:
                if len(movies_with_genres_list) > len(movies_with_cast_list):
                    smallest_list = movies_with_cast_list
                    biggest_list = movies_with_genres_list
                else:
                    smallest_list = movies_with_genres_list
                    biggest_list = movies_with_cast_list
                j = 0
                movies_blended_list = []
                for i in range(len(smallest_list)):
                    movies_blended_list.append(smallest_list[i])
                    movies_blended_list.append(biggest_list[i])
                    j = i
                for k in range(j, len(biggest_list)):
                    movies_blended_list.append(biggest_list[k])
                return movies_blended_list
            else:
                return movies_with_genres_list
        else:
            return movies_with_cast_list

    def get_genre_id(self, genre_name):
        return {
            'action': 28,
            'adventure': 12,
            'animation': 16,
            'comedy': 35,
            'comedies': 35,
            'crime': 80,
            'documentary': 99,
            'drama': 18,
            'family': 10751,
            'fantasy': 14,
            'history': 36,
            'horror': 27,
            'music': 10402,
            'romance': 10749,
            'romantic': 10749,
            'sci-fi': 878,
            'syfy': 878,
            'thriller': 53,
            'war': 10752,
            'western': 37
        }.get(genre_name, 0)

    def get_cast_id(self, cast_name):
        cast_name = cast_name.replace(" ", "%20")
        query_url = config.MOVIEDB_SEARCH_PERSON_ADDRESS + config.MOVIEDB_KEY + "&query=" + cast_name
        data = urllib.request.urlopen(query_url)
        result = data.read()
        movies = json.loads(result)
        return int(movies['results'][0]['id'])

    def set_movie_info(self, movie_name):
        movie_name = movie_name.replace(" ", "%20")
        omdbURL = config.OMDB_SEARCH_MOVIE_INFO + movie_name + "&r=json" + "&apikey=" + config.OMDB_KEY
        data = urllib.request.urlopen(omdbURL)
        result = data.read()
        movie_info = json.loads(result)
        self.movie['plot'] = movie_info.get("Plot")
        self.movie['actors'] = movie_info.get("Actors")
        self.movie['genres'] = movie_info.get("Genre")

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
