import ray
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from zoo_envs.env.marl_gwsh import MARL_GWSH
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
from omegaconf import OmegaConf

TRIALS = 10
TIME_STEPS_PER_TRIAL = 150

AGENT_LIST = ["agent_0", "agent_1", "agent_2"]

def eval_metrics(model, env_creator, env_config):
    reward_tally = {key: 0 for key in AGENT_LIST}
    stag_tally = {key: 0 for key in AGENT_LIST}
    plant_tally = {key: 0 for key in AGENT_LIST}

    env = env_creator(env_config)

    for trial in range(TRIALS):
        obs, info = env.reset()

        state_0 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]
        state_1 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]
        state_2 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]

        for time_step in range(TIME_STEPS_PER_TRIAL):
            result_0 = model.compute_single_action(obs["agent_0"], state_0, policy_id="agent_0")
            result_1 = model.compute_single_action(obs["agent_1"], state_1, policy_id="agent_1")
            result_2 = model.compute_single_action(obs["agent_2"], state_2, policy_id="agent_2")

            actions = {"agent_0": result_0[0], "agent_1": result_1[0], "agent_2": result_2[0]}
            state_0 = result_0[1]
            state_1 = result_1[1]
            state_2 = result_2[1]
            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in AGENT_LIST:
                stag_tally[agent] += 1 if infos[agent].get('capture', None) == 'stag' else 0
                stag_tally[agent] += 1 if infos[agent].get('capture', None) == 'poisoned_stag' else 0
                plant_tally[agent] += 1 if infos[agent].get('capture', None) == 'plant' else 0
                if infos[agent].get('capture', None) is not None:
                    reward_tally[agent] += rewards[agent]

    reward_count = {key: stag_tally[key] + plant_tally[key] for key in AGENT_LIST}

    # return {
    #     "mean_rewards": {key: value / reward_count[key] for key, value in reward_tally.items()},
    #     "stag_percentage": {key: value / reward_count[key] for key, value in stag_tally.items()},
    #     "plant_percentage": {key: value / reward_count[key] for key, value in plant_tally.items()}
    # }

    # return a list of all the values
    return [
        value / reward_count[key] if reward_count[key] != 0 else 0
        for key, value in reward_tally.items()
    ] + [
        value / reward_count[key] if reward_count[key] != 0 else 0
        for key, value in stag_tally.items()
    ] + [
        value / reward_count[key] if reward_count[key] != 0 else 0
        for key, value in plant_tally.items()
    ]


def default():

    reward_tally = {key: 0 for key in AGENT_LIST}
    stag_tally = {key: 0 for key in AGENT_LIST}
    plant_tally = {key: 0 for key in AGENT_LIST}

    ray.init()

    with open('cloud_checkpoint/config.yaml', 'r') as file:
        env_config = OmegaConf.load(file)['env']

    stag_reward = env_config['stag_reward']
    plant_reward = env_config['plant_reward']

    env_creator = lambda config: MARL_GWSH(**config)

    register_env(
        "MARL_GWSH",
        lambda config: ParallelPettingZooEnv(env_creator(env_config)),
    )

    env = env_creator(env_config)

    model = Algorithm.from_checkpoint('cloud_checkpoint')

    for trial in range(TRIALS):
        obs, info = env.reset()

        state_0 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]
        state_1 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]
        state_2 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]

        for time_step in range(TIME_STEPS_PER_TRIAL):
            result_0 = model.compute_single_action(obs["agent_0"], state_0, policy_id="agent_0")
            result_1 = model.compute_single_action(obs["agent_1"], state_1, policy_id="agent_1")
            result_2 = model.compute_single_action(obs["agent_2"], state_2, policy_id="agent_2")

            actions = {"agent_0": result_0[0], "agent_1": result_1[0], "agent_2": result_2[0]}
            state_0 = result_0[1]
            state_1 = result_1[1]
            state_2 = result_2[1]
            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in AGENT_LIST:
                stag_tally[agent] += 1 if infos[agent].get('capture', None) == 'stag' else 0
                plant_tally[agent] += 1 if infos[agent].get('capture', None) == 'plant' else 0
                if infos[agent].get('capture', None) is not None:
                    reward_tally[agent] += rewards[agent]

    reward_count = {key: stag_tally[key] + plant_tally[key] for key in AGENT_LIST}

    print("Mean rewards:")
    for key, value in reward_tally.items():
        print(f"{key}: {value / reward_count[key]:.2f}")

    print("\nStag percentage:")
    for key, value in stag_tally.items():
        print(f"{key}: {value / reward_count[key]:.2f}")

    print("\nPlant percentage:")
    for key, value in plant_tally.items():
        print(f"{key}: {value / reward_count[key]:.2f}")

    print({
        "mean_rewards": {key: value / reward_count[key] for key, value in reward_tally.items()},
        "stag_percentage": {key: value / reward_count[key] for key, value in stag_tally.items()},
        "plant_percentage": {key: value / reward_count[key] for key, value in plant_tally.items()}
    })

    ray.shutdown()

if __name__ == "__main__":
    default()