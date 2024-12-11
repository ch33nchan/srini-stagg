import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the existing data
data = pd.read_csv('collaboration_results/analysis.csv')

# Function to check for convergence with adjustable parameters
def check_convergence(rewards, threshold=0.01, window=10):
    if len(rewards) < window:
        return False, 0
    recent_rewards = rewards[-window:]
    return np.all(np.abs(np.diff(rewards[-window:])) < threshold), len(rewards)

# Analyze convergence for each agent with adjusted parameters
convergence_results = []
for agent in ["agent_0", "agent_1", "agent_2"]:
    rewards = data[f'avg_reward_{agent}'].values
    converged, epochs_to_converge = check_convergence(rewards, threshold=0.01, window=10)
    convergence_results.append((agent, converged, epochs_to_converge))
    print(f"Agent {agent}: Converged = {converged}, Epochs to Converge = {epochs_to_converge}")

# Plot the rewards to visualize convergence
plt.figure(figsize=(10, 6))
for agent in ["agent_0", "agent_1", "agent_2"]:
    plt.plot(data['poison_probability'], data[f'avg_reward_{agent}'], label=f'Agent {agent}')
plt.xlabel('Disease Probability')
plt.ylabel('Average Reward')
plt.title('Average Rewards vs Disease Probability')
plt.legend()
plt.grid(True)
plt.savefig('collaboration_results/average_rewards.png')
plt.show()
