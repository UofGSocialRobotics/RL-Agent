
CS_LABELS = ["SD", "PR", "HE", "VSN", "NONE"]
CS_LABELS_DICT = {'SD': [], 'VSN': [], 'PR': [], 'HE': [], 'NONE': []}

# User related
USER_LIST_ACTORS = "./resources/user/user_model/actor2id.lexicon"
USER_LIST_GENRES = "./resources/user/user_model/genre2id.lexicon"
USER_LIST_DIRECTORS = "./resources/user/user_model/director2id.lexicon"
USER_ACTIONS = "./resources/user/user_actions.lexicon"
USER_SENTENCES = "./resources/user/user_sentence_DB.csv"
USER_VOICE = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN - US_DAVID_11.0"

# Todo Learn Probas from data
PROBA_NO_ACTOR = [0.2, 0.8]
PROBA_NO_DIRECTOR = [0.2, 0.8]
PROBA_NO_GENRE = [0.2, 0.8]
# Todo Make sure it matches with actual actions
ITEMS_REQUEST_AFTER_MOVIE = ['yes', 'no', 'request(genre)', 'request(actor)', 'request(plot)', 'request(another)', 'inform(watched)']
PROBA_REQUEST_AFTER_MOVIE = [0.15, 0.1, 0.1, 0.1, 0.2, 0.2, 0.15]
PROBA_NUMBER_MOVIES = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
ITEMS_USER_TYPE = ["P", "I"]
PROBA_USER_TYPE = [0.641, 0.359]

# Agent related
AGENT_SENTENCES = "./resources/agent/agent_sentence_DB.csv"
AGENT_ACKS = "./resources/agent/ack_db.csv"
DM_MODEL = "./resources/agent/model.csv"
AGENT_VOICE = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
USE_ACKS = True

# Other
MOVIEDB_KEY = "6e3c2d4a2501c86cd7e0571ada291f55"
MOVIEDB_SEARCH_MOVIE_ADDRESS = "https://api.themoviedb.org/3/discover/movie?api_key="
MOVIE_DB_PROPERTY = "&sort_by=popularity.desc"
MOVIEDB_SEARCH_PERSON_ADDRESS = "https://api.themoviedb.org/3/search/person?api_key="
MOVIEDB_POSTER_PATH = "https://image.tmdb.org/t/p/original/"
OMDB_SEARCH_MOVIE_INFO = "http://www.omdbapi.com/?t="
OMDB_KEY = "be72fd68"

GENERATE_SENTENCE = True
GENERATE_VOICE = False

TRAINING_DIALOGUE_PATH = "./resources/training_dialogues/"
