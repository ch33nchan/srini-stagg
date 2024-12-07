import numpy as np
from typing import List, Dict, Any
import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from omegaconf import OmegaConf
from zoo_envs.env.marl_gwsh import MARL_GWSH

# Constants
AGENT_LIST = ["agent_0", "agent_1", "agent_2"]
TRIALS = 50
TIME_STEPS_PER_TRIAL = 150

def eval_metrics(model: Algorithm, env_creator: callable, env_config: Dict[str, Any]) -> List[float]:
    """
    Evaluate metrics for the trained model.
    
    Args:
        model: Trained RLlib algorithm
        env_creator: Function to create environment
        env_config: Environment configuration dictionary
    
    Returns:
        List of metrics in order: [rewards, stag_percentages, plant_percentages] for each agent
    """
    try:
        # Initialize tallies for each agent
        reward_tally = {key: 0 for key in AGENT_LIST}
        stag_tally = {key: 0 for key in AGENT_LIST}
        plant_tally = {key: 0 for key in AGENT_LIST}

        # Create environment
        env = env_creator(env_config)

        # Run evaluation trials
        for trial in range(TRIALS):
            obs, info = env.reset()

            # Initialize states for each agent
            states = {
                agent: [np.zeros(64, np.float32), np.zeros(64, np.float32)]
                for agent in AGENT_LIST
            }

            # Run steps for each trial
            for _ in range(TIME_STEPS_PER_TRIAL):
                # Compute actions for each agent
                results = {
                    agent: model.compute_single_action(
                        obs[agent], 
                        states[agent], 
                        policy_id=agent
                    )
                    for agent in AGENT_LIST
                }

                # Extract actions and update states
                actions = {agent: results[agent][0] for agent in AGENT_LIST}
                states = {agent: results[agent][1] for agent in AGENT_LIST}

                # Step environment
                obs, rewards, terminations, truncations, infos = env.step(actions)

                # Update tallies
                for agent in AGENT_LIST:
                    capture = infos[agent].get('capture', None)
                    if capture == 'stag' or capture == 'poisoned_stag':
                        stag_tally[agent] += 1
                    elif capture == 'plant':
                        plant_tally[agent] += 1
                    
                    if capture is not None:
                        reward_tally[agent] += rewards[agent]

                # Check for episode termination
                if any(terminations.values()) or any(truncations.values()):
                    break

        # Calculate total captures for each agent
        reward_count = {
            key: stag_tally[key] + plant_tally[key] 
            for key in AGENT_LIST
        }

        # Compile metrics list
        metrics = []
        for key in AGENT_LIST:
            # Add mean rewards
            metrics.append(
                reward_tally[key] / reward_count[key] 
                if reward_count[key] != 0 else 0
            )
        
        for key in AGENT_LIST:
            # Add stag percentages
            metrics.append(
                stag_tally[key] / reward_count[key] 
                if reward_count[key] != 0 else 0
            )
            
        for key in AGENT_LIST:
            # Add plant percentages
            metrics.append(
                plant_tally[key] / reward_count[key] 
                if reward_count[key] != 0 else 0
            )

        return metrics

    except Exception as e:
        print(f"Error in eval_metrics: {e}")
        # Return zeros if evaluation fails
        return [0.0] * (len(AGENT_LIST) * 3)

def default():
    """Default evaluation function for testing"""
    try:
        ray.init()

        # Load configuration
        with open('cloud_checkpoint/config.yaml', 'r') as file:
            env_config = OmegaConf.load(file)['env']

        # Create environment
        env_creator = lambda config: MARL_GWSH(**config)
        register_env(
            "MARL_GWSH",
            lambda config: ParallelPettingZooEnv(env_creator(env_config)),
        )

        # Load model
        model = Algorithm.from_checkpoint('cloud_checkpoint')

        # Run evaluation
        metrics = eval_metrics(model, env_creator, env_config)

        # Process results
        n_agents = len(AGENT_LIST)
        results = {
            "mean_rewards": {
                agent: metrics[i] 
                for i, agent in enumerate(AGENT_LIST)
            },
            "stag_percentage": {
                agent: metrics[i + n_agents] 
                for i, agent in enumerate(AGENT_LIST)
            },
            "plant_percentage": {
                agent: metrics[i + 2*n_agents] 
                for i, agent in enumerate(AGENT_LIST)
            }
        }

        # Print results
        print("\nMean rewards:")
        for agent, value in results["mean_rewards"].items():
            print(f"{agent}: {value:.2f}")

        print("\nStag percentage:")
        for agent, value in results["stag_percentage"].items():
            print(f"{agent}: {value:.2f}")

        print("\nPlant percentage:")
        for agent, value in results["plant_percentage"].items():
            print(f"{agent}: {value:.2f}")

        print("\nFull results:")
        print(results)

    except Exception as e:
        print(f"Error in default evaluation: {e}")
    
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    default()