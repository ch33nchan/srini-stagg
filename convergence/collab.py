import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the existing data
data = pd.read_csv('collaboration_results/analysis.csv')

# Function to check for convergence with adjustable parameters
def check_convergence(rewards, threshold, window):
    if len(rewards) < window:
        return False, 0
    recent_rewards = rewards[-window:]
    return np.all(np.abs(np.diff(recent_rewards)) < threshold), len(rewards)

# Define ranges for thresholds, window sizes, and epochs
thresholds = [0.001, 0.005, 0.01, 0.02]
window_sizes = [5, 10, 15, 20]
epochs = list(range(10, 101))  # Check from 10 to 100 epochs

# Analyze convergence for each agent with different parameters
converged_flag = False
for agent in ["agent_0", "agent_1", "agent_2"]:
    rewards = data[f'avg_reward_{agent}'].values
    for threshold in thresholds:
        for window in window_sizes:
            for epoch in epochs:
                if epoch <= len(rewards):
                    converged, epochs_to_converge = check_convergence(rewards[:epoch], threshold, window)
                    if converged:
                        print(f"Agent {agent}: Converged = {converged}, Threshold = {threshold}, Window = {window}, Epochs = {epoch}")
                        converged_flag = True

# Plot the rewards to visualize convergence only if converged
if converged_flag:
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
