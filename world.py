import user_sim
import rule_based_agent
import random
import utils

def main():
    user = user_sim.UserSimulator()
    agent = rule_based_agent.Agent()

    agent_action = {'intent': "start", 'movie': None}

    while agent_action['intent'] not in "goodbye":

        # print user sentence
        user_action = user.next(agent_action)
        user_action_to_say = random.choice(user.sentenceDB[user_action['intent']])
        print("U: " + utils.generate_entity_related_sentence(user_action_to_say, user_action['entity']))

        # print agent sentence
        agent_action = agent.next(user_action)
        agent_action_to_say = random.choice(agent.sentenceDB[agent_action['intent']])
        print("A: " + utils.generate_movie_related_sentence(agent_action_to_say, agent_action['movie']))

if __name__ == '__main__':
    main()


