import ray
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from zoo_envs.env.marl_gwsh import MARL_GWSH
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
from omegaconf import OmegaConf
import pygame
import imageio

ray.init()

# env_config = {
#     "number_of_agents": 3,
#     "max_steps": 50,
#     "observation_mode": "poison",
#     "stag_respawn": True,
#     "plant_respawn": True,
#     "reset_simplified": False,
#     "stag_poisoned_chance": 0.6,
#     "knows_poisoning": ['agent_2'],
#     # "reset_configs": {
#     #         'stag_positions': [(0, 0)],
#     #         'plant_positions': [(4, 4)],
#     #         'agent_positions': {'agent_0': (0, 4), 'agent_1': (0, 3), 'agent_2': (0, 2)},
#     # }
# }

with open('cloud_checkpoint/config.yaml', 'r') as file:
    env_config = OmegaConf.load(file)['env']

env_creator = lambda config: MARL_GWSH(**config)

register_env(
    "MARL_GWSH",
    lambda config: ParallelPettingZooEnv(env_creator(env_config)),
)

env_config["render_mode"] = "human"
env = env_creator(env_config)

model = Algorithm.from_checkpoint('cloud_checkpoint')

obs, info = env.reset()

state_0 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]
state_1 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]
state_2 = [np.zeros(64, np.float32), np.zeros(64, np.float32)]

# For gif generation
frames = []

for i in range(50):
    result_0 = model.compute_single_action(obs["agent_0"], state_0, policy_id="agent_0")
    result_1 = model.compute_single_action(obs["agent_1"], state_1, policy_id="agent_1")
    result_2 = model.compute_single_action(obs["agent_2"], state_2, policy_id="agent_2")

    actions = {"agent_0": result_0[0], "agent_1": result_1[0], "agent_2": result_2[0]}
    state_0 = result_0[1]
    state_1 = result_1[1]
    state_2 = result_2[1]

    obs, rewards, terminations, truncations, info = env.step(actions)

    print({"rewards": rewards, "terminations": terminations, "truncations": truncations, "info": info})

    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = np.rot90(frame, k=-1)
    frames.append(frame)

    pygame.time.wait(100)

imageio.mimsave('cloud_checkpoint/animation.gif', frames, duration=500)
pygame.quit()