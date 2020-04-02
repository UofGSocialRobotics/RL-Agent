import copy
import csv
import os
import deep_rl_agent
import qlearning_agent
import user_sim
import rule_based_agent
import config
import utils
import state_tracker
import ml_models
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def main():

    #agent_deep_rl = deep_rl_agent.Agent()
    agent_deep_rl = qlearning_agent.Agent()
    agent_rule_based = rule_based_agent.Agent()
    user = user_sim.UserSimulator()
    dst = state_tracker.DialogState()
    fig, subplots = plt.subplots(1, 3, sharex=True, sharey=True)

    print("Start")
    #pretrain_rewards, pretrain_task_rewards, pretrain_social_rewards = pretrain(agent_deep_rl, user, dst)
    #rl_agent_rewards, rl_task_rewards, rl_social_rewards = deep_rl_training(agent_deep_rl, user, dst)
    rule_based_rewards, rule_based_task_rewards, rule_based_social_rewards = rule_based_interactions(agent_rule_based,user, dst,agent_deep_rl)
    print("Rule based interactions done")
    rl_agent_rewards, rl_task_rewards, rl_social_rewards = deep_rl_training(agent_deep_rl, user, dst)
    print("Deep RL training done")
    utils.plotting_rewards("Total Rewards", rl_agent_rewards, rule_based_rewards, subplots, 0)
    utils.plotting_rewards("Task Rewards", rl_task_rewards, rule_based_task_rewards, subplots, 1)
    utils.plotting_rewards("Social Rewards", rl_social_rewards, rule_based_social_rewards, subplots, 2)
    plt.show()

def initialize(user,dst):
    user.generate_user()
    dst.set_initial_state()
    if config.VERBOSE_TRAINING > 1:
        print("Interaction number " + str(i) + "\nReco-Type: " + user.user_reco_pref + " --- Social-Type: " + user.user_type + " --- Recos Wanted: " + str(user.number_recos))
    agent_action = config.AGENT_ACTION
    user_action = config.USER_ACTION
    return agent_action, user_action

def pretrain(agent, state,sample_batch_size, action_encoder, user_action_encoder):
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
                    vectorized_action, vectorized_state = state.encode_state(action_encoder,user_action_encoder)
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

                        vectorized_action, vectorized_state = state.encode_state(action_encoder, user_action_encoder)

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
    #return list_rewards, list_task_rewards, list_social_rewards


def queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward):
    if i % config.EPISODES_THRESHOLD == 0:
        agent_reward_list.append(total_reward_agent / config.EPISODES_THRESHOLD)
        total_reward_agent = 0
    else:
        total_reward_agent += reward

    return agent_reward_list, total_reward_agent

def deep_rl_training(agent,user,dst):
    sample_batch_size = 90
    total_reward_agent = 0
    total_task_reward = 0
    total_social_reward = 0
    epsilon = config.EPSILON
    # list of rewards for the rl_agent across config.EPISODES episodes
    agent_reward_list = []
    agent_task_reward_list = []
    agent_social_reward_list = []

    print("-- Encoding action spaces")
    action_encoder, encoded_action_space = agent.get_action_space_encoder()
    user_action_encoder = user.get_action_encoder()
    print("-- Encoding State Space")
    agent_action, state = dst.encode_state(action_encoder, user_action_encoder)
    #agent.model = agent.build_DQN_model(len(state), agent_action.toarray().shape[1])
    print("-- Encoding done")

    pretrain(agent, dst, sample_batch_size, action_encoder, user_action_encoder)

    for i in tqdm(range(0, config.EPISODES)):
        if i > (config.EPISODES - config.EPISODES_THRESHOLD) and config.VERBOSE_TRAINING > -1:
            print("Tour " + str(i))
        agent_action, user_action = initialize(user, dst)
        vectorized_action, vectorized_state = dst.encode_state(action_encoder, user_action_encoder)

        while not dst.dialog_done and dst.turns < config.MAX_STEPS:
            vectorized_previous_state = copy.deepcopy(vectorized_state)
            agent_previous_action = copy.deepcopy(agent_action)
            user_previous_action = copy.deepcopy(user_action)

            agent_action = agent.next(dst)

            user_action = user.next(agent_action, dst)

            if i > (config.EPISODES - config.EPISODES_THRESHOLD) and config.VERBOSE_TRAINING > -1:
                print("A: ", agent_action)
                print("U: ", user_action)

            dst.update_state(agent_action, user_action, agent_previous_action, user_previous_action)
            reward, task_reward, rapport_reward = compute_reward(dst)
            agent.update_qtables(vectorized_previous_state, vectorized_state, agent_action, reward)
            vectorized_action, vectorized_state = dst.encode_state(action_encoder, user_action_encoder)
            #agent.remember(vectorized_previous_state, vectorized_action, reward, vectorized_state, dst.dialog_done)

        #agent.replay(sample_batch_size)
        agent_reward_list, total_reward_agent = queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent,reward)
        agent_task_reward_list, total_task_reward = queue_rewards_for_plotting(i, agent_task_reward_list,total_task_reward, task_reward)
        agent_social_reward_list, total_social_reward = queue_rewards_for_plotting(i, agent_social_reward_list,total_social_reward, rapport_reward)

        if epsilon >= config.EPSILON_MIN:
            epsilon *= config.EPSILON_DECAY

    #agent.save_model()
    agent.qtable.to_csv(config.QTABLE)
    print(agent.qtable)
    print("epsilon", str(epsilon))

    return agent_reward_list, agent_task_reward_list, agent_social_reward_list

def rule_based_interactions(agent, user, dst, rl_agent):
    total_reward_agent = 0
    total_task_reward = 0
    total_social_reward = 0
    agent_reward_list = []# list of rewards for the rule_based_agent across config.EPISODES episodes
    agent_task_reward_list = []
    agent_social_reward_list = []

    print("-- Encoding action spaces")
    action_encoder, encoded_action_space = rl_agent.get_action_space_encoder()
    user_action_encoder = user.get_action_encoder()

    for i in tqdm(range(0, config.EPISODES)):
        agent.init_agent()
        agent_action, user_action = initialize(user, dst)
        vectorized_action, vectorized_state = dst.encode_state(action_encoder, user_action_encoder)

        while not dst.dialog_done and dst.turns < config.MAX_STEPS:
            agent_previous_action = copy.deepcopy(agent_action)
            user_previous_action = copy.deepcopy(user_action)
            vectorized_previous_state = copy.deepcopy(vectorized_state)

            agent_action = agent.next(dst)

            user_action = user.next(agent_action, dst)

            if config.VERBOSE_TRAINING > 1:
                print("A: ", agent_action)
                print("U: ", user_action)

            dst.update_state(agent_action, user_action, agent_previous_action, user_previous_action)
            reward, task_reward, rapport_reward = compute_reward(dst)
            rl_agent.update_qtables(vectorized_previous_state, vectorized_state, agent_action, reward)
            vectorized_action, vectorized_state = dst.encode_state(action_encoder, user_action_encoder)

        agent_reward_list, total_reward_agent = queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward)
        agent_task_reward_list, total_task_reward = queue_rewards_for_plotting(i, agent_task_reward_list, total_task_reward, task_reward)
        agent_social_reward_list, total_social_reward = queue_rewards_for_plotting(i, agent_social_reward_list, total_social_reward, rapport_reward)

    print(rl_agent.qtable)
    return agent_reward_list, agent_task_reward_list, agent_social_reward_list


def compute_reward(state):
    task_reward = 0
    rapport_reward = 0
    reward = 0
    #Todo do not say bye before user gets to his limit

    #####################       Task Reward     #########################
    reward += -1
    agent_action = state.state["current_agent_action"]
    if "request" in agent_action['intent'] and all(item in state.state["slots_filled"] for item in ['genre','actor','director']):
        reward += -10
    if "introduce" in agent_action['intent'] and all(item in state.state["slots_filled"] for item in ['role','reason_like','last_movie']):
        reward += -10
    if "greeting" in agent_action['intent'] and state.turns > 1:
        reward += -10
    if "introduce" in agent_action['intent'] and len(state.state["slots_filled"]) > 0:
        reward += -10
    #if "last_movie" in agent_action['intent'] and "last_movie" in state.state["slots_filled"]:
    #    reward += -30
    if state.dialog_done:
        #reward -= state.turns
        if state.delivered_recos != 0:
            #print("user wanted " + str(state.delivered_recos) + " recos and accepted " + str(state.accepted_recos))
            reward = reward + (state.accepted_recos/state.delivered_recos) * 100
            task_reward = reward

    #####################       Social Reward     #########################
            data = []
            data.extend(state.rec_I_user)
            data.extend(state.rec_I_agent)
            rapport = ml_models.estimate_rapport(data)
            rapport_reward = ml_models.get_rapport_reward(rapport, state.rec_P_agent / state.turns)
            reward = reward + rapport_reward[0]
            #print("Rapport :" + str(rapport))
            #print("Reward total: " + str(self.reward))
            #print("Reward from Rapport: " + str(rapport_reward) + " and from Task: " + str(task_reward))
    return reward, task_reward, rapport_reward

if __name__ == '__main__':
    #utils.unpickle_dialogues(config.RAW_DIALOGUES_PATH + "*_full_dialog.pkl")
    #ml_models.build_reciprocity_dataset()
    #utils.preprocess_dialogue_data()
    main()


