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

    state =

    data = []

    while agent_action['intent'] not in "goodbye":
        agent_action = agent.next(user_action)
        utils.generate_agent_sentence(agent, agent_action, agent_previous_action, user_action)

        user_action = user.next(agent_action)
        utils.generate_user_sentence(user, user_action, agent_action)

        rec_I_user, rec_I_agent, rec_P_agent = re.append_data_from_simulation(user_action, agent_action, agent_previous_action, rec_I_user, rec_I_agent, rec_P_agent, user.user_type)

        agent_previous_action = agent_action
        turns += 1

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


