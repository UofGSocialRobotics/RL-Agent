import copy
import random

import user_sim
import rule_based_agent
import rl_agent
import simplified_rl_agent
import utils
import state_tracker
import numpy as np
import matplotlib.pyplot as plt

def main_rule_based():
    user = user_sim.UserSimulator()
    agent = rule_based_agent.Agent()
    print(user.number_recos)
    print(user.user_type)

    agent_action = agent_previous_action = {'intent': "start", 'movie': None, 'ack_cs': None, 'cs': None}

    while agent_action['intent'] not in "goodbye":
        user_action = user.next(agent_action)
        utils.generate_user_sentence(user, user_action, agent_action)

        agent_action = agent.next(user_action)
        utils.generate_agent_sentence(agent, agent_action, agent_previous_action, user_action)

        agent_previous_action = agent_action


def main_rl():

    agent = simplified_rl_agent.Agent()
    episodes = 10000  # Amount of games
    max_steps = 20  # Maximum steps per episode
    epsilon = 1.0  # Greed 100%
    epsilon_min = 0.005  # Minimum greed 0.05%
    epsilon_decay = 0.993  # Decay multiplied with epsilon after each episode
    reward_list = []

    for i in range(0, episodes):
        user = user_sim.UserSimulator()
        dst = state_tracker.DialogState()
        dst.set_initial_state(user)
        print("Interaction number " + str(i) + "\nReco-Type: " + user.user_reco_pref + " --- Social-Type: " + user.user_type + " --- Recos Wanted: " + str(user.number_recos))
        agent_action = {'intent': 'start', 'ack_cs': '', 'cs': ''}
        user_action = {'intent': '', 'cs': '', 'entity': '', 'entity_type': '', 'polarity': ''}

        while not dst.dialog_done and dst.turns < max_steps:
            state = copy.deepcopy(dst.state)
            agent_previous_action = copy.deepcopy(agent_action)
            user_previous_action = copy.deepcopy(user_action)

            if random.uniform(0, 1) > epsilon:
                agent_action = agent.next_best(dst.state)
            else:
                agent_action = agent.next()

            user_action = user.next(agent_action, dst.state)

            if i > 9950:
                print("A: ", agent_action)
                print("U: ", user_action)

            dst.update_state(agent_action, user_action, agent_previous_action, user_previous_action, user.user_type)
            reward = dst.compute_reward(state, agent_action)
            agent.update_qtables(state, dst.state, agent_action, reward)

        if i > 9950:
            print("final reward: ", str(reward))
        reward_list.append(reward)
        if epsilon >= epsilon_min:
            epsilon *= epsilon_decay
    #print(agent.task_qtable)
    plt.plot(reward_list)
    plt.ylabel('reward')
    plt.show()

if __name__ == '__main__':
    main_rl()
    #main_rule_based()


