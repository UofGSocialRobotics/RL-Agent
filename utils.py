from itertools import zip_longest
import os
import csv
import pyttsx3
import random
import win32com.client as wincl
import config
import urllib.request
import json
from sklearn.preprocessing import OneHotEncoder
import pickle
import glob
import numpy


def unpickle_dialogues(files):
    txt_files = glob.glob(files)
    cs_SARA_count = 0
    cs_User_count = 0
    cs_ack_count = 0
    cs_count_filename = config.RAW_DIALOGUES_PATH + "cs_count_1.csv"
    cs_count_file = open(cs_count_filename, "w")

    for file in txt_files:
        dialog_done = False
        file_tab = file.split("\\")
        filename = config.TRAINING_DIALOGUE_PATH + file_tab[1] + ".csv"
        tosave = open(filename, "w")
        pickle_off = open(file, "rb")
        emp = pickle.load(pickle_off)
        for key, value in emp.items():
            id_tab = file.split('\\')
            id_tab2 = id_tab[1].split('_')
            if 'introduce' in emp[key]['SARA']['other']['ts']['intent']:
                emp[key]['SARA']['other']['ts']['slot'] = 'role'
            if 'request' in emp[key]['SARA']['other']['ts']['intent']:
                if emp[key]['SARA']['other']['ts']['slot'] in ['last_movie', 'reason_like']:
                    emp[key]['SARA']['other']['ts']['intent'] = 'introduce'
                elif emp[key]['SARA']['other']['ts']['slot'] in 'feedback':
                    emp[key]['SARA']['other']['ts']['intent'] = 'bye'
                    emp[key]['SARA']['other']['ts']['slot'] = 'feedback'
                elif emp[key]['SARA']['other']['ts']['slot'] in 'another_one':
                     emp[key]['SARA']['other']['ts']['intent'] = 'another_one'
                     emp[key]['SARA']['other']['ts']['slot'] = ''
                elif emp[key]['SARA']['other']['ts']['slot'] in 'reason_not_like':
                     emp[key]['SARA']['other']['ts']['intent'] = 'reason_not_like'
                     emp[key]['SARA']['other']['ts']['slot'] = ''
            if not dialog_done:
                toprint = "1," + id_tab2[0] + ",SARA, ack," + emp[key]['SARA']['ack']['cs'] + "," + emp[key]['SARA']['other']['ts'][
            'intent'] + "," + emp[key]['SARA']['other']['ts']['slot'] + "," + emp[key]['SARA']['other'][
                      'cs'] + ", USER, " + emp[key]['USER']['ts']['intent'] + "," + emp[key]['USER']['ts'][
                      'slot'] + "," + emp[key]['USER']['cs'] + "\n"
                tosave.write(toprint)
                if (emp[key]['SARA']['other']['cs'] != 'NONE'):
                    cs_SARA_count += 1
                if (emp[key]['SARA']['ack']['cs'] != ''):
                    cs_ack_count += 1
                if (emp[key]['USER']['cs'] != ''):
                    cs_User_count += 1
                if  'bye' in emp[key]['USER']['ts']['intent']:
                    dialog_done = True

        tosave.close()
        cs_count = str(cs_ack_count) + "," + str(cs_SARA_count) + "," + str(cs_User_count) + "\n"
        cs_count_file.write(cs_count)
        cs_SARA_count = 0
        cs_User_count = 0
        cs_ack_count = 0

    cs_count_file.close()

def write_interactions_file(turns, interactions_file):
    file = open(interactions_file, "w")
    for action in turns:
        file.writelines(str(action) + "\n")
    file.close()

def transform_agent_action(action_dict):
    action = []
    action.append(action_dict['ack_cs'])
    action.append(action_dict['intent'])
    action.append(action_dict['cs'])
    return(action)
    #return numpy.array(action).reshape(1, -1)

def transform_user_action(action_dict):
    action = []
    action.append(action_dict['intent'].replace(' ',''))
    action.append(action_dict['entity_type'])
    action.append(action_dict['cs'])
    return(action)
    #return numpy.array(action).reshape(1, -1)

def queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward):
    if i % config.EPISODES_THRESHOLD == 0:
        agent_reward_list.append(total_reward_agent / config.EPISODES_THRESHOLD)
        total_reward_agent = 0
    else:
        total_reward_agent += reward

    return agent_reward_list, total_reward_agent

def preprocess_dialogue_data():
    print("Creating lexicons and NN data... ")
    dialogue_path = config.TRAINING_DIALOGUE_PATH
    agent_actions_file = config.AGENT_ACTIONS
    agent_intentions_file = config.AGENT_INTENTIONS
    agent_action_space = config.AGENT_ACTION_SPACE
    user_action_space = config.USER_ACTION_SPACE
    slots_file = config.SLOTS
    user_actions_file = config.USER_ACTIONS
    agent_actions_list = []
    agent_intent_list = []
    slots = []
    user_actions_list = []
    all_user_types = []
    all_agent_ack_CS = []
    all_agent_actions = []
    all_agent_CS = []
    all_user_actions = []
    all_user_CS = []

    triple = ""
    triple_list = []
    double = ""
    user_triple = ""
    user_triple_list = []
    ack_cs_lexicon = []
    cs_lexicon = []

    for root, dirs, files in os.walk("./resources/training_dialogues"):
        for name in files:
            if name.endswith(".pkl.csv"):
                with open(dialogue_path + "/" + name, mode='rt') as csv_file:
                    interaction = csv.reader(csv_file, delimiter=',')
                    for row in interaction:
                        if row[0] in [1,2,3]:
                            all_user_types.append("I")
                        else:
                            all_user_types.append("P")
                        if row[4]:
                            all_agent_ack_CS.append(row[4])
                        else:
                            all_agent_ack_CS.append('0')
                        all_agent_actions.append(row[5])
                        all_agent_CS.append(row[7])
                        all_user_actions.append(row[9])
                        all_user_CS.append(row[11])

                        #Create_lexicons
                        if row[5] not in agent_intent_list:
                            agent_intent_list.append(row[5])
                        if row[6] not in slots and row[6] != '':
                            slots.append(row[6])
                        if row[9] not in user_actions_list:
                            user_actions_list.append(row[9])
                        triple = str(row[4]) + "," + str(row[5]) + "," + str(row[7])
                        if triple not in triple_list:
                            triple_list.append(triple)
                        double = str(str(row[5]) + "," + str(row[6]))
                        if double not in agent_actions_list:
                            agent_actions_list.append(double)
                        user_triple = str(row[9]).replace(" ","") + "," + str(row[10])
                        if user_triple not in user_triple_list:
                            user_triple_list.append(user_triple)
                        if row[4] not in ack_cs_lexicon:
                            ack_cs_lexicon.append(row[4])
                        if row[7] not in cs_lexicon:
                            cs_lexicon.append(row[7])

    file_agent = open(agent_actions_file, "w")
    for action in agent_actions_list:
        file_agent.writelines(action + "\n")
    file_agent.close()

    file_intents = open(agent_intentions_file, "w")
    for intent in agent_intent_list:
        file_intents.writelines(intent + "\n")
    file_intents.close()

    file_slots = open(slots_file, "w")
    for slot in slots:
        file_slots.writelines(slot + "\n")
    file_slots.close()

    file_user = open(user_actions_file, "w")
    for action in user_actions_list:
        file_user.write(action + "\n")
    file_user.close()

    file_action_space = open(agent_action_space, "w")
    file_action_space.writelines(",," + "\n")
    for action in triple_list:
        file_action_space.write(str(action) + "\n")
    file_action_space.close()

    file_user_action_space = open(user_action_space, "w")
    file_user_action_space.writelines(",," + "\n")
    for action in user_triple_list:
        for cs in ["HE","PR","SD","QESD","NONE"]:
            tmp_action = action + "," + cs
            file_user_action_space.write(tmp_action + "\n")
    file_user_action_space.close()

    print("Lexicons created")

def encode(data):
    enc = OneHotEncoder()
    return enc.fit(data)

def parse_intention(intent):
    tab = intent.split('(')
    intention =  tab[0]
    entity_type = tab[1].replace(')','')
    return intention, entity_type



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



def plotting_rewards(title, rl_rewards, rule_based_rewards, subplots, col):
    subplots[col].title.set_text(title)
    subplots[col].plot(rl_rewards, label="rl_reward")
    subplots[col].plot(rule_based_rewards, label="rule_based_reward")
    subplots[col].legend(loc='best')



#################################################################################################################
#################################################################################################################
##############                                                                                ###################
##############                                NLG Related Functions                           ###################
##############                                                                                ###################
#################################################################################################################
#################################################################################################################

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


def pretrain(agent, state,action_encoder, user_action_encoder):
    print("Pretraining using data ")
    dialogue_path = config.TRAINING_DIALOGUE_PATH
    list_rewards = []
    list_task_rewards = []
    list_social_rewards = []
    rapport_scores = pd.read_csv(config.RAPPORT_GROUPS, header=None, names=['id', 'group', 'rapport'])
    cpt = 0

    for root, dirs, files in os.walk("./resources/training_dialogues"):
        for name in tqdm(files):
            if name.endswith(".pkl.csv"):
                total_rewards = 0
                task_rewards = 0
                social_rewards = 0
                with open(dialogue_path + "/" + name, mode='rt') as csv_file:
                    interaction = csv.reader(csv_file, delimiter=',')
                    user_current_action = config.USER_ACTION
                    agent_current_action = config.AGENT_ACTION
                    vectorized_action, vectorized_state = state.get_state()
                    for row in interaction:

                        cpt += 1
                        agent_previous_action = copy.deepcopy(agent_current_action)
                        user_previous_action = copy.deepcopy(user_current_action)
                        vectorized_previous_state = copy.deepcopy(vectorized_state)

                        agent_current_action['ack_cs'] = row[4]
                        agent_current_action['intent'] = row[5]
                        agent_current_action['entity_type'] = row[6]
                        agent_current_action['cs'] = row[7]

                        user_current_action['intent'] = row[9].replace(" ","")
                        user_current_action['entity_type'] = row[10]
                        user_current_action['cs'] = row[11]
                        state.update_state(agent_current_action, user_current_action, agent_previous_action, user_previous_action)
                        total_reward, task_reward, social_reward = compute_reward(state)
                        agent.update_qtables(vectorized_previous_state, vectorized_state, agent_current_action, total_reward)
                        total_rewards += total_reward
                        task_rewards += task_reward
                        social_rewards += social_reward

                        vectorized_action, vectorized_state = state.get_state()

                        #agent.remember(vectorized_state, vectorized_action, total_rewards, vectorized_previous_state, state.dialog_done)
                        #agent.replay(sample_batch_size)

                    list_rewards.append(total_rewards)
                    list_task_rewards.append(task_rewards)
                    social_reward = rapport_scores.loc[rapport_scores['id'].isin([row[1]])]
                    list_social_rewards.append((social_reward.iloc[0,2]/7)*100)
                    state.set_initial_state()

    print("Pretraining done ")
    print(list_rewards)
    print(agent.qtable)
    agent.qtable.to_csv(config.QTABLE)
    list_rewards[0] = 0
    list_task_rewards[0] = 0
    return list_rewards, list_task_rewards, list_social_rewards


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

