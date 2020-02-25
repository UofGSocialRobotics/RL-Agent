import copy
import random
import deep_rl_agent
import user_sim
import rule_based_agent
import config
import simplified_rl_agent
import utils
import state_tracker
import matplotlib.pyplot as plt

def main():

    agent_rl = simplified_rl_agent.Agent()
    agent_deep_rl = deep_rl_agent.Agent()
    agent_rule_based = rule_based_agent.Agent()
    user = user_sim.UserSimulator()
    dst = state_tracker.DialogState()
    fig, subplots = plt.subplots(1, 3, sharex=True, sharey=True)

    print("Start")
    rl_agent_rewards, rl_task_rewards, rl_social_rewards = deep_rl_training(agent_deep_rl, user, dst)
    print("Deep RL training done")
    #rl_agent_rewards, rl_task_rewards, rl_social_rewards = rl_training(agent_rl, user, dst)
    #print("RL training done")
    rule_based_rewards, rule_based_task_rewards, rule_based_social_rewards = rule_based_interactions(agent_rule_based, user, dst)
    print("Rule based interactions done")
    utils.plotting_rewards("Total Rewards", rl_agent_rewards, rule_based_rewards, subplots, 0)
    utils.plotting_rewards("Task Rewards", rl_task_rewards, rule_based_task_rewards, subplots, 1)
    utils.plotting_rewards("Social Rewards", rl_social_rewards, rule_based_social_rewards, subplots, 2)
    plt.show()

def initialize(user,dst):
    user.generate_user()
    dst.set_initial_state(user)
    if config.VERBOSE_TRAINING > 1:
        print("Interaction number " + str(i) + "\nReco-Type: " + user.user_reco_pref + " --- Social-Type: " + user.user_type + " --- Recos Wanted: " + str(user.number_recos))
    agent_action = {'intent': 'start', 'ack_cs': '', 'cs': ''}
    user_action = {'intent': '', 'cs': '', 'entity': '', 'entity_type': '', 'polarity': ''}
    return agent_action, user_action

def queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward):
    if i % config.EPISODES_THRESHOLD == 0:
        agent_reward_list.append(total_reward_agent / config.EPISODES_THRESHOLD)
        total_reward_agent = 0
    else:
        total_reward_agent += reward

    return agent_reward_list, total_reward_agent

def deep_rl_training(agent,user,dst):
    sample_batch_size = 32
    total_reward_agent = 0
    total_task_reward = 0
    total_social_reward = 0
    epsilon = config.EPSILON
    agent_reward_list = []  # list of rewards for the rl_agent across config.EPISODES episodes
    agent_task_reward_list = []
    agent_social_reward_list = []

    for i in range(0, config.EPISODES):
        print("Tour " + str(i))
        agent_action, user_action = initialize(user, dst)

        while not dst.dialog_done and dst.turns < config.MAX_STEPS:
            state = copy.deepcopy(dst.state)
            vectorized_state = dst.vectorize()
            agent_previous_action = copy.deepcopy(agent_action)
            user_previous_action = copy.deepcopy(user_action)

            agent_action = agent.next(dst)

            user_action = user.next(agent_action, dst)

            if i > (config.EPISODES - config.EPISODES_THRESHOLD) and config.VERBOSE_TRAINING > 0:
                print("A: ", agent_action)
                print("U: ", user_action)

            dst.update_state(agent_action, user_action, agent_previous_action, user_previous_action, user.user_type)
            reward, task_reward, rapport_reward = dst.compute_reward(state, agent_action, user.number_recos)
            vectorized_action = agent.vectorize_action(agent_action)
            agent.remember(vectorized_state, vectorized_action, reward, dst.vectorize(), dst.dialog_done)
            agent.update_qtables(state, dst.state, agent_action, agent_previous_action, user_action,
                                 user_previous_action, reward)

        agent.replay(sample_batch_size)
        agent_reward_list, total_reward_agent = queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent,
                                                                           reward)
        agent_task_reward_list, total_task_reward = queue_rewards_for_plotting(i, agent_task_reward_list,
                                                                               total_task_reward, task_reward)
        agent_social_reward_list, total_social_reward = queue_rewards_for_plotting(i, agent_social_reward_list,
                                                                                   total_social_reward, rapport_reward)

        if epsilon >= config.EPSILON_MIN:
            epsilon *= config.EPSILON_DECAY

    agent.save_model()
    print("epsilon", str(epsilon))
    #print(agent.task_qtable)
    #agent.task_qtable.to_csv(config.TASK_QTABLE, encoding='utf-8')
    #agent.social_qtable.to_csv(config.SOCIAL_QTABLE, encoding='utf-8')

    return agent_reward_list, agent_task_reward_list, agent_social_reward_list

def rl_training(agent, user, dst):
    total_reward_agent = 0
    total_task_reward = 0
    total_social_reward = 0
    epsilon = config.EPSILON
    agent_reward_list = []  # list of rewards for the rl_agent across config.EPISODES episodes
    agent_task_reward_list = []
    agent_social_reward_list = []

    for i in range(0, config.EPISODES):
        agent_action, user_action = initialize(user, dst)

        while not dst.dialog_done and dst.turns < config.MAX_STEPS:
            state = copy.deepcopy(dst.state)
            agent_previous_action = copy.deepcopy(agent_action)
            user_previous_action = copy.deepcopy(user_action)

            if random.uniform(0, 1) > epsilon:
                agent_action = agent.next_best(dst.state, user_action)
            else:
                agent_action = agent.next()

            user_action = user.next(agent_action, dst)

            if i > (config.EPISODES - config.EPISODES_THRESHOLD) and config.VERBOSE_TRAINING > 0:
                print("A: ", agent_action)
                print("U: ", user_action)

            dst.update_state(agent_action, user_action, agent_previous_action, user_previous_action, user.user_type)
            reward, task_reward, rapport_reward = dst.compute_reward(state, agent_action, user.number_recos)
            agent.update_qtables(state, dst.state, agent_action, agent_previous_action, user_action,
                                    user_previous_action, reward)

        agent_reward_list, total_reward_agent = queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward)
        agent_task_reward_list, total_task_reward = queue_rewards_for_plotting(i, agent_task_reward_list, total_task_reward, task_reward)
        agent_social_reward_list, total_social_reward = queue_rewards_for_plotting(i, agent_social_reward_list, total_social_reward, rapport_reward)


        if epsilon >= config.EPSILON_MIN:
            epsilon *= config.EPSILON_DECAY

    print("epsilon", str(epsilon))
    print(agent.task_qtable)
    agent.task_qtable.to_csv(config.TASK_QTABLE, encoding='utf-8')
    agent.social_qtable.to_csv(config.SOCIAL_QTABLE, encoding='utf-8')

    return agent_reward_list, agent_task_reward_list, agent_social_reward_list


def rule_based_interactions(agent, user, dst):
    total_reward_agent = 0
    total_task_reward = 0
    total_social_reward = 0
    agent_reward_list = []# list of rewards for the rule_based_agent across config.EPISODES episodes
    agent_task_reward_list = []
    agent_social_reward_list = []

    for i in range(0, config.EPISODES):
        agent.init_agent()
        agent_action, user_action = initialize(user, dst)

        while not dst.dialog_done and dst.turns < config.MAX_STEPS:
            state = copy.deepcopy(dst.state)
            agent_previous_action = copy.deepcopy(agent_action)
            user_previous_action = copy.deepcopy(user_action)

            agent_action = agent.next(user_action)

            user_action = user.next(agent_action, dst)

            if config.VERBOSE_TRAINING > 1:
                print("A: ", agent_action)
                print("U: ", user_action)

            dst.update_state(agent_action, user_action, agent_previous_action, user_previous_action, user.user_type)
            reward, task_reward, rapport_reward = dst.compute_reward(state, agent_action, user.number_recos)

        agent_reward_list, total_reward_agent = queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward)
        agent_task_reward_list, total_task_reward = queue_rewards_for_plotting(i, agent_task_reward_list, total_task_reward, task_reward)
        agent_social_reward_list, total_social_reward = queue_rewards_for_plotting(i, agent_social_reward_list, total_social_reward, rapport_reward)

    return agent_reward_list, agent_task_reward_list, agent_social_reward_list


if __name__ == '__main__':
    utils.unpickle_dialogues(config.RAW_DIALOGUES_PATH + "*_full_dialog.pkl")
    #utils.preprocess_dialogue_data()
    #main()


