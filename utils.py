from itertools import zip_longest

import pyttsx3
import random
import win32com.client as wincl
import config
import urllib.request
import json
import matplotlib.pyplot as plt


def set_voice_engine(who, voice):
    if "U" in who:
        engine = wincl.Dispatch("SAPI.SpVoice")
        return engine
    else:
        engine = pyttsx3.init()
        engine.setProperty('voice', voice)
        return engine

def load_agent_ack_model(path):
    ackDB = {}
    with open(path) as f:
        for line in f:
            line_input = line.split(",")
            if ackDB.get(line_input[0]) is None:
                yes_cs_dict = {'SD': [], 'VSN': [], 'PR': [], 'HE': [], 'NONE': []}
                no_cs_dict = {'SD': [], 'VSN': [], 'PR': [], 'HE': [], 'NONE': []}
                default_cs_dict = {'SD': [], 'VSN': [], 'PR': [], 'HE': [], 'NONE': []}
                ackDB[line_input[0]] = {'yes': yes_cs_dict, 'no': no_cs_dict, 'default': default_cs_dict}
            ackDB[line_input[0]][line_input[4]][line_input[2]].append(line_input[3])
    return ackDB

def load_agent_sentence_model(path):
    sentenceDB = {}
    with open(path) as f:
        for line in f:
            line_input = line.split(",")
            if sentenceDB.get(line_input[0]) is not None:
                sentenceDB.get(line_input[0]).append(line_input[2])
            else:
                sentenceDB[line_input[0]] = [line_input[2]]
    return sentenceDB


def load_user_sentence_model(path):
    sentenceDB = {}
    with open(path) as f:
        for line in f:
            line_input = line.split(",")
            if sentenceDB.get(line_input[0]) is None:
                sentenceDB[line_input[0]] = {'SD': [], 'VSN': [], 'PR': [], 'HE': [], 'NONE': []}
            sentenceDB[line_input[0]][line_input[1]].append(line_input[2])
    return sentenceDB


def generate_user_sentence(user, user_action, agent_action):
    new_user_action = select_specific_action(user_action, agent_action)
    if user.sentenceDB[new_user_action['intent']][new_user_action['cs']]:
        sentence = random.choice(user.sentenceDB[new_user_action['intent']][new_user_action['cs']])
    else:
        sentence = random.choice(user.sentenceDB[new_user_action['intent']]['NONE'])
    sentence_to_say = replace_in_user_sentence(sentence, new_user_action, agent_action)
    print("U: " + sentence_to_say)
    if config.GENERATE_VOICE:
        user.engine.Speak(sentence_to_say)


def generate_agent_sentence(agent, agent_action, agent_prev_action, user_action):
    if config.USE_ACKS:
        sentence = select_ack(agent, agent_prev_action, user_action)
    if sentence != "":
        sentence = sentence + " " + random.choice(agent.sentenceDB[agent_action['intent']])
    else:
        sentence = random.choice(agent.sentenceDB[agent_action['intent']])
    sentence_to_say = replace_in_agent_sentence(sentence, agent_action['movie'], user_action['entity'])
    sentence_to_say = sentence_to_say.replace("–", "-")
    print("A: " + sentence_to_say)
    if config.GENERATE_VOICE:
        agent.engine.say(sentence_to_say)
        agent.engine.runAndWait()


def select_ack(agent, agent_prev_action, user_action):
    if agent_prev_action['intent'] in agent.ackDB:
        if "yes" in user_action['intent'] and agent.ackDB[agent_prev_action['intent']]:
            if user_action['cs'] in agent.ackDB[agent_prev_action['intent']]['yes']:
                ack = pick_ack(agent, agent_prev_action['intent'], 'yes', user_action)
            else:
                ack = pick_ack(agent, agent_prev_action['intent'], 'yes', 'NONE')
        elif "no" in user_action['intent'] and agent.ackDB[agent_prev_action['intent']]:
            if user_action['cs'] in agent.ackDB[agent_prev_action['intent']]['no']:
                ack = pick_ack(agent, agent_prev_action['intent'], 'no', user_action)
            else:
                ack = pick_ack(agent, agent_prev_action['intent'], 'no', 'NONE')
        else:
            if agent.ackDB[agent_prev_action['intent']]['default']:
                if user_action['cs'] in agent.ackDB[agent_prev_action['intent']]['default']:
                    ack = pick_ack(agent, agent_prev_action['intent'], 'default', user_action)
                else:
                    ack = pick_ack(agent, agent_prev_action['intent'], 'default', 'NONE')
            else:
                ack = ""
    else:
        ack = ""
    return ack


def pick_ack(agent, agent_prev_action, valence, user_action):
    potential_options = []
    #print(agent.ackDB[agent_prev_action][valence], user_action['cs'])
    for option in agent.ackDB[agent_prev_action][valence][user_action['cs']]:
        if "#entity" in option:
            if user_action['entity']:
                potential_options.append(option)
        else:
            potential_options.append(option)
    #print(str(potential_options) + " pour l'action " + agent_prev_action)
    if potential_options:
        return random.choice(potential_options)
    else:
        return ""


def select_specific_action(user_action, agent_action):
    new_action = dict(user_action)
    if "greet" in agent_action['intent']:
        if "yes" in new_action['intent']:
            new_action['intent'] = 'positive_greeting'
        else:
            new_action['intent'] = 'negative_greeting'
    if "yes" in new_action['intent']:
        if '(another)' in agent_action['intent']:
            new_action['intent'] = 'yes(another)'
        else: #'(movie)' in agent_action['intent']:
            new_action['intent'] = 'yes(movie)'
    elif "no" in new_action['intent']:
        if "request(genre)" in agent_action['intent']:
            new_action['intent'] = 'inform(genre=no)'
        elif "request(actor)" in agent_action['intent']:
            new_action['intent'] = 'inform(actor=no)'
        elif "request(director)" in agent_action['intent']:
            new_action['intent'] = 'inform(director=no)'
        elif "request(another)" in agent_action['intent']:
            new_action['intent'] = 'no(another)'
    return new_action

def replace_in_user_sentence(sentence, user_action, agent_action):
    if "#entity" in sentence:
        sentence = sentence.replace("#entity", user_action['entity'])
    return sentence


def replace_in_agent_sentence(sentence, movie, entity):
    if "#title" in sentence:
        sentence = sentence.replace("#title", movie['title'])
    if "#plot" in sentence:
        if movie['plot']:
            sentence = sentence.replace("#plot", movie['plot'])
        else:
            sentence = "Sorry, I have no idea what this movie is about..."
    if "#actors" in sentence:
        if movie['actors']:
            sentence = sentence.replace("#actors", movie['actors'])
        else:
            sentence = "Sorry, I don't remember who plays in this one..."
    if "#genres" in sentence:
        if movie['genres']:
            sentence = sentence.replace("#genres", movie['genres'])
        else:
            sentence = "Sorry, I'm not sure about this movie's genres..."
    if "#entity" in sentence:
        sentence = sentence.replace("#entity", entity)
    return sentence

def plotting_rewards(rl_rewards, rule_based_rewards):
    plt.plot(rl_rewards, label="rl_reward")
    plt.plot(rule_based_rewards, label="rule_based_reward")
    plt.ylabel('reward')
    plt.xlabel('# of iterations')
    plt.legend(loc='best')
    plt.show()

#################################################################################################################
#################################################################################################################
##############                                                                                ###################
##############                       Movie recommendation functions                           ###################
##############                                                                                ###################
#################################################################################################################
#################################################################################################################

def query_blended_movies_list(user_model):
    final_list = []
    if user_model['liked_genres']:
        genre_id = get_genre_id(user_model['liked_genres'][-1].lower())
    if user_model['liked_cast']:
        cast_id = get_person_id(user_model['liked_cast'][-1].lower())
    if user_model['liked_crew']:
        crew_id = get_person_id(user_model['liked_crew'][-1].lower())

    if user_model['liked_genres'] and user_model['liked_cast'] and user_model['liked_crew']:
        query1 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_cast=" + str(cast_id) + "&with_crew=" + str(crew_id) + config.MOVIE_DB_PROPERTY
        query2 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(genre_id) + "&with_people=" + str(cast_id) + "," + str(crew_id) + config.MOVIE_DB_PROPERTY
        query3 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(genre_id) + config.MOVIE_DB_PROPERTY
        query4 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_people=" + str(cast_id) + "," + str(crew_id) + config.MOVIE_DB_PROPERTY
        list1 = get_movie_list(query1)
        list2 = get_movie_list(query2)
        list3 = get_movie_list(query3)
        list4 = get_movie_list(query4)
        for list in [list1,list2, list3, list4]:
            if not list:
                list = None
        final_list = [y for x in zip_longest(list1, list2, list3, list4, fillvalue=None) for y in x if y is not None]
    elif user_model['liked_cast'] and user_model['liked_crew']:
        query1 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_cast=" + str(
            cast_id) + "&with_crew=" + str(crew_id) + config.MOVIE_DB_PROPERTY
        query2 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_people=" + str(
            cast_id) + "," + str(crew_id) + config.MOVIE_DB_PROPERTY
        list1 = get_movie_list(query1)
        list2 = get_movie_list(query2)
        final_list = [y for x in zip_longest(list1, list2, fillvalue=None) for y in x if y is not None]
    elif user_model['liked_cast'] and user_model['liked_genres']:
        query1 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(
            genre_id) + "&with_cast=" + str(cast_id) + config.MOVIE_DB_PROPERTY
        query2 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(
            genre_id) + config.MOVIE_DB_PROPERTY
        query3 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_cast=" + str(
            cast_id) + config.MOVIE_DB_PROPERTY
        final_list = get_movie_list(query1)
        list2 = get_movie_list(query2)
        list3 = get_movie_list(query3)
        final_list.extend([y for x in zip_longest(list2, list3, fillvalue=None) for y in x if y is not None])
    elif user_model['liked_genres'] and user_model['liked_crew']:
        query1 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(
            genre_id) + "&with_crew=" + str(crew_id) + config.MOVIE_DB_PROPERTY
        query2 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(
            genre_id) + config.MOVIE_DB_PROPERTY
        query3 = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_crew=" + str(
            crew_id) + config.MOVIE_DB_PROPERTY
        final_list = get_movie_list(query1)
        list2 = get_movie_list(query2)
        list3 = get_movie_list(query3)
        final_list.extend([y for x in zip_longest(list2, list3, fillvalue=None) for y in x if y is not None])
    elif user_model['liked_cast']:
        query = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_cast=" + str(
            cast_id) + config.MOVIE_DB_PROPERTY
        final_list = get_movie_list(query)
    elif user_model['liked_crew']:
        query = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_crew=" + str(
            crew_id) + config.MOVIE_DB_PROPERTY
        final_list = get_movie_list(query)
    elif user_model['liked_genres']:
        query = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + "&with_genres=" + str(
            genre_id) + config.MOVIE_DB_PROPERTY
        final_list = get_movie_list(query)

    query_url = config.MOVIEDB_SEARCH_MOVIE_ADDRESS + config.MOVIEDB_KEY + config.MOVIE_DB_PROPERTY
    final_list.extend(get_movie_list(query_url))
    return final_list


def get_movie_list(query):
    data = urllib.request.urlopen(query)
    result = data.read()
    movies = json.loads(result)
    return movies['results']


def get_genre_id(genre_name):
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
        'scifi': 878,
        'syfy': 878,
        'thriller': 53,
        'war': 10752,
        'western': 37
    }.get(genre_name, 0)


def get_person_id(cast_name):
    cast_name = cast_name.replace(" ", "%20")
    query_url = config.MOVIEDB_SEARCH_PERSON_ADDRESS + config.MOVIEDB_KEY + "&query=" + cast_name
    data = urllib.request.urlopen(query_url)
    result = data.read()
    movies = json.loads(result)
    return int(movies['results'][0]['id'])


def get_movie_info(movie_name):
    movie_name = movie_name.replace(" ", "%20")
    movie_name = movie_name.replace("é", "e")
    movie_name = movie_name.replace("–", "-")
    movie_name = movie_name.encode('ascii', 'ignore').decode('ascii')
    omdbURL = config.OMDB_SEARCH_MOVIE_INFO + movie_name + "&r=json" + "&apikey=" + config.OMDB_KEY
    data = urllib.request.urlopen(omdbURL)
    result = data.read()
    movie_info = json.loads(result)
    return movie_info

