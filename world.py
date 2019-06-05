import user_sim
import rule_based_agent

def main():
    user = user_sim.UserSimulator()
    agent = rule_based_agent.Agent()
    print("User-Type: " + user.user_type)
    print("Preferences: " + user.pref_director + " - " + user.pref_genre + " - " + user.pref_actor)
    print("Number of Recos: " + str(user.number_recos))

    agent_action = {'intent': "start", 'movie': None}

    while agent_action['intent'] not in "goodbye":
        user_action = user.next(agent_action)
        if user_action['entity']:
            print("U: " + user_action['intent'] + " likes " + user_action['entity'])
        else:
            print("U: " + user_action['intent'])
        agent_action = agent.next(user_action)
        print("A: " + agent_action['intent'] + "   " + agent_action['movie']['title'])

if __name__ == '__main__':
    main()

