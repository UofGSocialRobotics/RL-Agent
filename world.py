import copy
import rl_agent
import user_sim
import config
import utils
import state_tracker
import ml_models
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():

    agent = rl_agent.Agent()
    user = user_sim.UserSimulator()
    dst = state_tracker.DialogState()
    fig, subplots = plt.subplots(1, 3, sharex=True, sharey=True)
    plt.setp(subplots[0], ylabel='Accumulated rewards')
    plt.setp(subplots[1], xlabel='Epochs / 100')

    print("Start")
    rule_based_rewards, rule_based_task_rewards, rule_based_social_rewards = run_interactions(agent, user, dst, 0)
    print("Rule based interactions done")
    rl_agent_rewards, rl_task_rewards, rl_social_rewards = run_interactions(agent, user, dst, 1)
    print("Unimodal RL training done")
    bimodal_rl_agent_rewards, bimodal_rl_task_rewards, bimodal_rl_social_rewards = run_interactions(agent, user, dst, 2)
    print("Bimodal RL training done")

    utils.plotting_rewards("Total Rewards", rl_agent_rewards, bimodal_rl_agent_rewards, rule_based_rewards, subplots, 0)
    utils.plotting_rewards("Task Rewards", rl_task_rewards, bimodal_rl_task_rewards, rule_based_task_rewards, subplots, 1)
    utils.plotting_rewards("Social Rewards", rl_social_rewards, bimodal_rl_social_rewards, rule_based_social_rewards, subplots, 2)
    plt.show()

def initialize(user,dst, i, turns):
    user.generate_user()
    dst.set_initial_state(user)
    if config.VERBOSE_TRAINING > -1:
        turns.append("Interaction number " + str(i) + "\nReco-Type: " + user.user_reco_pref + " --- Social-Type: " + user.user_type + " --- Recos Wanted: " + str(user.number_recos))
    agent_action = config.AGENT_ACTION
    user_action = config.USER_ACTION
    return agent_action, user_action

def run_interactions(agent, user, dst, mode):
    turns = []
    total_reward_agent = 0
    total_task_reward = 0
    epsilon = config.EPSILON
    total_social_reward = 0
    agent_reward_list = []# list of rewards for the rule_based_agent across config.EPISODES episodes
    agent_task_reward_list = []
    agent_social_reward_list = []

    for i in tqdm(range(0, config.EPISODES)):
        agent.init_agent()
        agent_action, user_action = initialize(user, dst, i, turns)
        state = dst.get_state()

        while not dst.dialog_done and dst.turns < config.MAX_STEPS:
            agent_previous_action = copy.deepcopy(agent_action)
            user_previous_action = copy.deepcopy(user_action)
            previous_state = copy.deepcopy(state)

            if mode == 0:
                agent_action = agent.next(dst)
            else:
                agent_action = agent.next_rl(dst, epsilon)
            user_action = user.next(agent_action, dst)

            previous_dst = copy.deepcopy(dst)
            dst.update_state(agent_action, user_action, agent_previous_action, user_previous_action)
            reward, task_reward, rapport_reward = compute_reward(dst, previous_dst, mode)
            state = dst.get_state()
            agent.update_qtables(previous_state, state, agent_action, reward)

            if config.VERBOSE_TRAINING > -1:
                turns.append(agent_action)
                turns.append(user_action)
                turns.append(reward)

        agent_reward_list, total_reward_agent = utils.queue_rewards_for_plotting(i, agent_reward_list, total_reward_agent, reward)
        agent_task_reward_list, total_task_reward = utils.queue_rewards_for_plotting(i, agent_task_reward_list, total_task_reward, task_reward)
        agent_social_reward_list, total_social_reward = utils.queue_rewards_for_plotting(i, agent_social_reward_list, total_social_reward, rapport_reward)

        if epsilon >= config.EPSILON_MIN and mode > 0:
            epsilon *= config.EPSILON_DECAY

    if config.VERBOSE_TRAINING > -1:
        if mode == 0:
            utils.write_interactions_file(turns, config.RULE_BASED_INTERACTIONS_FILE)
        elif mode == 1:
            utils.write_interactions_file(turns, config.RL_INTERACTIONS_FILE)
        else:
            utils.write_interactions_file(turns, config.RL_INTERACTIONS_FILE_BIMODAL)
    agent.qtable.to_csv(config.QTABLE)
    print(epsilon)
    return agent_reward_list, agent_task_reward_list, agent_social_reward_list


def compute_reward(state, previous_state, mode):
    task_reward = 0
    rapport_reward = 0
    reward = state.reward
    #Todo do not say bye before user gets to his limit

    #####################       Task Reward     #########################
    reward += -1
    agent_action = state.state["current_agent_action"]
    if "request" in agent_action['intent'] and all(item in previous_state.state["slots_filled"] for item in ['genre','actor','director']):
        reward += -10
    if "introduce" in agent_action['intent'] and all(item in previous_state.state["slots_filled"] for item in ['role','reason_like','last_movie']):
        reward += -10
    if "greeting" in agent_action['intent'] and state.turns > 1:
        reward += -10
    if "introduce" in agent_action['intent'] and len(previous_state.state["slots_filled"]) > 0:
        reward += -10
    if state.dialog_done:
        if state.delivered_recos == 0:
            reward = -100
        else:
            if state.accepted_recos != 0:
                reward = reward + (state.accepted_recos/state.delivered_recos) * 100
            else:
                reward = -100
            task_reward = reward

    #####################       Social Reward     #########################
            data = []
            data.extend(state.rec_I_user)
            data.extend(state.rec_I_agent)
            rapport = ml_models.estimate_rapport(data)
            rapport_reward = ml_models.get_rapport_reward(mode, state.user_type, rapport, state.rec_P_agent / (state.turns*2))
            reward = reward + rapport_reward#[0]
    state.reward = reward
    return reward, task_reward, rapport_reward

if __name__ == '__main__':
    #utils.unpickle_dialogues(config.RAW_DIALOGUES_PATH + "*_full_dialog.pkl")
    #utils.preprocess_dialogue_data()
    #ml_models.build_reciprocity_dataset()
    #ml_models.test_re()
    #ml_models.build_social_model()

    main()


