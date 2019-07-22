import user_sim
import rule_based_agent
import rl_agent
import utils
import rapport_estimator as re
import numpy as np

def main():
    user = user_sim.UserSimulator()
    #agent = rule_based_agent.Agent()
    agent = rl_agent.Agent()
    print(user.number_recos)
    print(user.user_type)

    agent_action = agent_previous_action = {'intent': "start", 'movie': None, 'ack_cs': None, 'cs': None}
    rec_I_user = [0, 0, 0, 0, 0]
    rec_I_agent = [0, 0, 0, 0, 0]
    rec_P_agent = 0
    turns = 0

    epsilon = 1.0  # Greed 100%
    epsilon_min = 0.005  # Minimum greed 0.05%
    epsilon_decay = 0.99993  # Decay multiplied with epsilon after each episode
    episodes = 50000  # Amount of games
    max_steps = 100  # Maximum steps per episode
    learning_rate = 0.65

    gamma = 0.65

    reward = 0

    data = []

    while agent_action['intent'] not in "goodbye":

        user_previous_action = user_action
        # print user sentence
        user_action = user.next(agent_action)
        utils.generate_user_sentence(user, user_action, agent_action)

        if turns > 0:
            # Update our Q-table with our Q-function
            agent.cs_qtable[user_previous_action['cs'], agent_action['cs']] = (1 - learning_rate) * agent.cs_qtable[user_previous_action['cs'], agent_action['cs']] \
                + learning_rate * (reward + gamma * np.max(agent.cs_qtable[user_action['cs'], :]))
        # print user sentence
        agent_action = agent.next(user_action)
        utils.generate_agent_sentence(agent, agent_action, agent_previous_action, user_action)

        agent_previous_action = agent_action
        turns += 1

        rec_I_user, rec_I_agent, rec_P_agent = re.append_data_from_simulation(user_action,agent_action, agent_previous_action, rec_I_user, rec_I_agent, rec_P_agent, user.user_type)


        # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
        #if epsilon >= epsilon_min:
            #epsilon *= epsilon_decay


    data.extend(rec_I_user)
    data.extend(rec_I_agent)
    re.estimate_rapport(data)
    reward = re.get_rapport_reward(re.estimate_rapport(data), rec_P_agent/turns, user.user_type)
    print("Rapport :" + str(re.estimate_rapport(data)))
    print("Reward :" + str(reward))
    print(agent.cs_qtable)

    agent.cs_qtable[user_previous_action['cs'], agent_action['cs']] = (1 - learning_rate) * agent.cs_qtable[
        user_previous_action['cs'], agent_action['cs']] \
                                                                      + learning_rate * (reward + gamma * np.max(
        agent.cs_qtable[user_action['cs'], :]))







if __name__ == '__main__':
    main()


