import user_sim
import rule_based_agent
import utils

def main():
    user = user_sim.UserSimulator()
    agent = rule_based_agent.Agent()
    print(user.number_recos)

    agent_action = agent_previous_action = {'intent': "start", 'movie': None}

    while agent_action['intent'] not in "goodbye":

        # print user sentence
        user_action = user.next(agent_action)
        utils.generate_user_sentence(user, user_action, agent_action)

        # print user sentence
        agent_action = agent.next(user_action)
        utils.generate_agent_sentence(agent, agent_action, agent_previous_action, user_action)
        agent_previous_action = agent_action


if __name__ == '__main__':
    main()


