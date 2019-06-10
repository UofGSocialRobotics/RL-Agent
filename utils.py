import pyttsx3
import random
import win32com.client as wincl
import config


def set_voice_engine(who, voice):
    if "U" in who:
        engine = wincl.Dispatch("SAPI.SpVoice")
        return engine
    else:
        engine = pyttsx3.init()
        engine.setProperty('voice', voice)
        return engine


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


def generate_agent_sentence(agent, action):
    sentence = random.choice(agent.sentenceDB[action['intent']])
    sentence_to_say = replace_in_agent_sentence(sentence, action['movie'])
    print("A: " + sentence_to_say)
    if config.GENERATE_VOICE:
        agent.engine.say(sentence_to_say)
        agent.engine.runAndWait()


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


def replace_in_agent_sentence(sentence, movie):
    if "#title" in sentence:
        # movListString = ""
        # for mov in self.moviesList:
        #     movListString = movListString + " " + mov
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
    return sentence
