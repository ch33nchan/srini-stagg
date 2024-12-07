import hydra
from omegaconf import DictConfig
import numpy as np
from zoo_envs.env.marl_gwsh import MARL_GWSH
import ray
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
import torch
import csv
import os
import matplotlib.pyplot as plt

def evaluate_collaboration(model, env_config, num_episodes=100):
    env = MARL_GWSH(**env_config)
    collaborations = 0
    total_rewards = {f"agent_{i}": 0 for i in range(env_config["number_of_agents"])}
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        states = {
            f"agent_{i}": [np.zeros(64, np.float32), np.zeros(64, np.float32)]
            for i in range(env_config["number_of_agents"])
        }
        done = False
        
        while not done:
            actions = {}
            for agent_id in env.agents:
                result = model.compute_single_action(
                    obs[agent_id], 
                    states[agent_id],
                    policy_id=agent_id
                )
                actions[agent_id] = result[0]
                states[agent_id] = result[1]
            
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent_id in rewards:
                total_rewards[agent_id] += rewards[agent_id]
                if infos[agent_id].get('capture', None) == 'stag':
                    collaborations += 1
                    
            done = any(terminations.values()) or any(truncations.values())
    
    avg_rewards = {k: v/num_episodes for k, v in total_rewards.items()}
    collab_rate = collaborations / (num_episodes * env_config["number_of_agents"])
    
    return collab_rate, avg_rewards

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Create results directory
    os.makedirs("collaboration_results", exist_ok=True)
    
    # Setup CSV logging
    results_file = "collaboration_results/analysis.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["poison_probability", "collaboration_rate", "avg_reward_agent_0", 
                        "avg_reward_agent_1", "avg_reward_agent_2"])
    
    # Test range of poison probabilities
    poison_probs = np.arange(0.0, 1.1, 0.1)
    results = []
    
    for prob in poison_probs:
        print(f"\nTraining with poison probability: {prob:.1f}")
        
        # Update environment config
        env_config = dict(cfg.env)
        env_config["stag_poisoned_chance"] = float(prob)
        
        # Initialize Ray and environment
        ray.init()
        env_creator = lambda config: MARL_GWSH(**config)
        register_env(
            "MARL_GWSH",
            lambda config: ParallelPettingZooEnv(env_creator(config))
        )
        
        # Setup policies
        policies = {
            f"agent_{i}" for i in range(env_config["number_of_agents"])
        }
        
        # Configure PPO with LSTM
        ppo_config = (PPOConfig()
            .environment(
                env="MARL_GWSH",
                env_config=env_config
            )
            .framework(framework="torch")
            .resources(num_gpus=1 if torch.cuda.is_available() else 0)
            .env_runners(num_env_runners=3)
            .training(
                train_batch_size=512,
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
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=lambda aid, *args, **kwargs: aid
            )
            .rl_module(
                model_config_dict={
                    "fcnet_hiddens": [128, 64],
                    "use_lstm": True,
                    "lstm_cell_size": 64,
                    "lstm_use_prev_action": True,
                    "lstm_use_prev_reward": True,
                    "vf_share_layers": False,
                    "soft_horizon": True,
                    "no_done_at_end": False,
                    "horizon": 200,
                },
                rl_module_spec=MultiAgentRLModuleSpec(
                    module_specs={p: SingleAgentRLModuleSpec() for p in policies},
                )
            )
        )
        
        # Train model
        ppo = ppo_config.build()
        
        for i in range(cfg.training.num_iterations):
            result = ppo.train()
            if i % 100 == 0:
                print(f"Training iteration {i}")
                reward_list = [
                    result["env_runners"]["policy_reward_mean"]["agent_0"],
                    result["env_runners"]["policy_reward_mean"]["agent_1"],
                    result["env_runners"]["policy_reward_mean"]["agent_2"]
                ]
                rounded_reward_list = [float("%.2f" % reward) for reward in reward_list]
                print(f"Current rewards: {rounded_reward_list}")
        
        # Evaluate collaboration
        collab_rate, avg_rewards = evaluate_collaboration(ppo, env_config)
        
        # Save results
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                prob, 
                collab_rate, 
                avg_rewards["agent_0"],
                avg_rewards["agent_1"],
                avg_rewards["agent_2"]
            ])
        
        results.append((prob, collab_rate, avg_rewards))
        
        # Cleanup
        ppo.stop()
        ray.shutdown()
    
    # Plot results
    plot_results(results)

def plot_results(results):
    probs = [r[0] for r in results]
    collab_rates = [r[1] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(probs, collab_rates, 'b-o')
    plt.xlabel('Poison Probability')
    plt.ylabel('Collaboration Rate')
    plt.title('Collaboration Rate vs Poison Probability')
    plt.grid(True)
    plt.savefig('collaboration_results/collaboration_rate.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for agent in ["agent_0", "agent_1", "agent_2"]:
        rewards = [r[2][agent] for r in results]
        plt.plot(probs, rewards, '-o', label=agent)
    plt.xlabel('Poison Probability')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards vs Poison Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('collaboration_results/average_rewards.png')
    plt.close()

if __name__ == "__main__":
    main()