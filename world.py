import copy
import random
import user_sim
import rule_based_agent
import config
import simplified_rl_agent
import utils
import state_tracker

def main():

    agent_rl = simplified_rl_agent.Agent()
    agent_rule_based = rule_based_agent.Agent()
    user = user_sim.UserSimulator()
    dst = state_tracker.DialogState()

    print("Start")
    rl_agent_rewards = rl_training(agent_rl, user, dst)
    print("RL training done")
    rule_based_rewards = rule_based_interactions(agent_rule_based, user, dst)
    print("Rule based interactions done")
    utils.plotting_rewards(rl_agent_rewards, rule_based_rewards)

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


def rl_training(agent, user, dst):
    total_reward_agent = 0
    epsilon = config.EPSILON
    agent_reward_list = []  # list of rewards for the rl_agent across config.EPISODES episodes

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
            reward = dst.compute_reward(state, agent_action, user.number_recos)
            agent.update_qtables(state, dst.state, agent_action, agent_previous_action, user_action,
                                    user_previous_action, reward)

        agent_reward_list, total_reward_agent = queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward)

        if epsilon >= config.EPSILON_MIN:
            epsilon *= config.EPSILON_DECAY

    # print("epsilon", str(epsilon))
    # print(agent.task_qtable)
    return agent_reward_list


def rule_based_interactions(agent, user, dst):
    total_reward_agent = 0
    agent_reward_list = []  # list of rewards for the rule_based_agent across config.EPISODES episodes
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
            reward = dst.compute_reward(state, agent_action, user.number_recos)

        agent_reward_list, total_reward_agent = queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward)

    return agent_reward_list


if __name__ == '__main__':
    main()


