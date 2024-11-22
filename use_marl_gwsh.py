import numpy as np
from zoo_envs.env.marl_gwsh import MARL_GWSH
from agent import Agent
from gymnasium.spaces import Discrete, Dict, Tuple, Sequence, MultiDiscrete
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="use_config")
def my_app(cfg: DictConfig) -> None:

    env = MARL_GWSH(**cfg.env)

    observations, infos = env.reset()
    env.render()

    print(env.observation_space("agent_0"))

    wasd_to_action = {
        'x': 0,
        'w': 1,
        's': 2,
        'a': 3,
        'd': 4
    }

    # agent1 = Agent("agent_0", MultiDiscrete([5, 5]), 1)

    while True:
        # ask for input
        actions = {}
        actions["agent_0"] = [wasd_to_action[input(f"Agent action: ")]][0]
        actions["agent_1"] = 0 # np.random.randint(5)
        actions["agent_2"] = 0 # np.random.randint(5)
        print("Actions: ", actions)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # The observations are flatted 5x5 grids concatenated together
        # Print them in 5x5 grids
        # print("Observations: ")
        # print(observations["agent_0"])
        for agent_id, obs in observations.items():
            obs = obs[:-1].reshape(4, 9, 9)
            print(agent_id)
            for row in obs:
                print(row.T)
        print("Rewards: ", rewards)
        if any(truncations.values()):
            break

    env.close()


if __name__ == "__main__":
    my_app()

