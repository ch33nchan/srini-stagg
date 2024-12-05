import hydra
from omegaconf import DictConfig, OmegaConf
import csv
import os
import time
from zoo_envs.env.marl_gwsh import MARL_GWSH
import ray
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig, PPO
import torch
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="custom_config")
def custom_training(cfg: DictConfig):
    env_config = cfg["env"]
    env_creator = lambda config: MARL_GWSH(**config)

    ray.init()
    register_env(
        "MARL_GWSH",
        lambda config: ParallelPettingZooEnv(env_creator(env_config)),
    )

    policies = {
        f"agent_{i}" for i in range(env_config["number_of_agents"])
    }

    num_gpu = 1 if torch.cuda.is_available() else 0

    ppo_config = PPOConfig().environment(env="MARL_GWSH").env_runners(num_env_runners=3).training(
        train_batch_size=128,
        lr=2.5e-4,
        gamma=0.99,
        lambda_=0.95,
        use_gae=True,
        clip_param=0.3,
        grad_clip=None,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        sgd_minibatch_size=64,
        num_sgd_iter=1,
    ).debugging(log_level="ERROR").framework(framework="torch").resources(num_gpus=num_gpu).multi_agent(
        policies=policies,
        policy_mapping_fn=(lambda aid, *args, **kwargs: aid)
    )

    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
    num_runs = 10
    results_dict = {prob: [] for prob in probabilities}

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    for prob in probabilities:
        env_config["stag_poisoned_chance"] = prob
        print(f"Training with probability: {prob}")
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            # Create a new PPO instance for each run
            ppo = ppo_config.build()
            
            # Train for a specified number of iterations
            for i in range(cfg["training"]["num_iterations"]):
                result = ppo.train()
                
                # Extract rewards for all agents
                if "env_runners" in result and "policy_reward_mean" in result["env_runners"]:
                    rewards = []
                    for agent_id in range(env_config["number_of_agents"]):
                        agent_key = f"agent_{agent_id}"
                        if agent_key in result["env_runners"]["policy_reward_mean"]:
                            rewards.append(result["env_runners"]["policy_reward_mean"][agent_key])
                    
                    if rewards:
                        mean_reward = np.mean(rewards)
                        results_dict[prob].append(mean_reward)
                
                if i % 10 == 0:  # Print progress every 10 iterations
                    print(f"Iteration {i}/{cfg['training']['num_iterations']}")
            
            # Stop the current PPO instance
            ppo.stop()

    # Calculate averages and save results
    with open('results/collab_rate_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['probability', 'average_reward', 'std_reward'])
        
        for prob in probabilities:
            if results_dict[prob]:
                avg_reward = np.mean(results_dict[prob])
                std_reward = np.std(results_dict[prob])
                writer.writerow([prob, avg_reward, std_reward])
                print(f"Probability {prob}: Average reward = {avg_reward:.4f}, Std = {std_reward:.4f}")
            else:
                print(f"Warning: No results collected for probability {prob}")

    ray.shutdown()

if __name__ == "__main__":
    custom_training()