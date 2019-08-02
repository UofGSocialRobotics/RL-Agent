import user_sim
import rule_based_agent
import rl_agent
import utils
import state_tracker
import rapport_estimator as re
import numpy as np

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
    user = user_sim.UserSimulator()
    agent = rl_agent.Agent()
    dst = state_tracker.DialogState()
    print(user.number_recos)
    print(user.user_type)

    agent_action = agent_previous_action = {'intent': "start", 'movie': None, 'ack_cs': None, 'cs': None}

    dst.set_initial_state(user)


    while agent_action['intent'] not in "goodbye":
        agent_action = agent.next(dst.state)
        print("A: ", agent_action)

        user_action = user.next(agent_action)
        #print("U: ", user_action)

        dst.update_state(agent_action, user_action)
        #print(dst.state)

        #rec_I_user, rec_I_agent, rec_P_agent = re.append_data_from_simulation(user_action, agent_action, agent_previous_action, rec_I_user, rec_I_agent, rec_P_agent, user.user_type)

        #agent_previous_action = agent_action

    #data.extend(rec_I_user)
    #data.extend(rec_I_agent)
    #re.estimate_rapport(data)
    #reward = re.get_rapport_reward(re.estimate_rapport(data), rec_P_agent/turns, user.user_type)
    #print("Rapport :" + str(re.estimate_rapport(data)))
    #print("Reward :" + str(reward))
    #print(agent.cs_qtable)


if __name__ == '__main__':
    main_rl()


