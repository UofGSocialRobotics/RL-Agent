import user_sim
import rule_based_agent
import utils
import rapport_estimator as re

def main():
    user = user_sim.UserSimulator()
    agent = rule_based_agent.Agent()
    print(user.number_recos)
    print(user.user_type)

    agent_action = agent_previous_action = {'intent': "start", 'movie': None, 'ack_cs': None, 'cs': None}
    rec_user = [0, 0, 0, 0, 0]
    rec_agent = [0, 0, 0, 0, 0]
    data = []

    while agent_action['intent'] not in "goodbye":

        # print user sentence
        user_action = user.next(agent_action)
        utils.generate_user_sentence(user, user_action, agent_action)

        # print user sentence
        agent_action = agent.next(user_action)
        utils.generate_agent_sentence(agent, agent_action, agent_previous_action, user_action)

        rec_user, rec_agent = re.append_data_from_simulation(user_action, agent_action, agent_previous_action, rec_user, rec_agent)

        agent_previous_action = agent_action

    data.extend(rec_user)
    data.extend(rec_agent)
    re.estimate_rapport(data)
    print("Rapport :" + str(re.estimate_rapport(data)))

if __name__ == '__main__':
    main()


