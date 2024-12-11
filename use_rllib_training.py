
import ray
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from zoo_envs.env.marl_gwsh import MARL_GWSH
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
from omegaconf import OmegaConf
import pygame
import imageio
import os

ray.init()

checkpoint_dirs = ["checkpoints_1", "checkpoints_2", "checkpoints_3", "checkpoints_4"]

for checkpoint_dir in checkpoint_dirs:
    # Load environment configuration
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, 'r') as file:
        env_config = OmegaConf.load(file)['env']

    env_creator = lambda config: MARL_GWSH(**config)

    register_env(
        "MARL_GWSH",
        lambda config: ParallelPettingZooEnv(env_creator(config)),
    )

    env_config["render_mode"] = "human"
    env = env_creator(env_config)

    model = Algorithm.from_checkpoint(checkpoint_dir)

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

    gif_path = os.path.join(checkpoint_dir, "animation.gif")
    imageio.mimsave(gif_path, frames, duration=500)
    print(f"Saved animation to: {gif_path}")
    pygame.quit()


ray.shutdown()
