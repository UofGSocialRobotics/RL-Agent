import copy
import user_sim
import rule_based_agent
import rl_agent
import simplified_rl_agent
import utils
import state_tracker
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

    agent = simplified_rl_agent.Agent()
    episodes = 1000  # Amount of games
    max_steps = 30  # Maximum steps per episode

    for i in range(0, episodes):
        user = user_sim.UserSimulator()
        dst = state_tracker.DialogState()
        dst.set_initial_state(user)
        print("Interaction number " + str(i) + "\nReco-Type: " + user.user_reco_pref + " --- Social-Type: " + user.user_type + " --- Recos Wanted: " + str(user.number_recos))

        while not dst.dialog_done and dst.turns < max_steps:
            state = copy.deepcopy(dst.state)
            if i < 995:
                agent_action = agent.next()
            else:
                agent_action = agent.next_best(dst.state)
                print("A: ", agent_action)

            user_action = user.next(agent_action)
            print("U: ", user_action)

            dst.update_state(agent_action, user_action)
            reward = dst.compute_simple_reward()

            if str(state) in agent.cs_qtable.index:
                if str(dst.state) not in agent.cs_qtable.index:
                    agent.cs_qtable.loc[str(dst.state)] = 0
                agent.cs_qtable.at[str(state), agent_action['intent']] = (1 - agent.learning_rate) * agent.cs_qtable.at[str(state), agent_action['intent']] + agent.learning_rate * (reward + agent.gamma * np.max(agent.cs_qtable.loc[str(dst.state), :]))
            else:
                agent.cs_qtable.loc[str(state)] = 0
                if str(dst.state) not in agent.cs_qtable.index:
                    agent.cs_qtable.loc[str(dst.state)] = 0
                agent.cs_qtable.at[str(state), agent_action['intent']] = (1 - agent.learning_rate) * agent.cs_qtable.at[str(state), agent_action['intent']] + agent.learning_rate * (reward + agent.gamma * np.max(agent.cs_qtable.loc[str(dst.state), :]))

    print(agent.cs_qtable)

if __name__ == '__main__':
    main_rl()
    #main_rule_based()


