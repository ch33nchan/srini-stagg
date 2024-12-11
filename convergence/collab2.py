import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the existing data
data = pd.read_csv('collaboration_results/analysis.csv')
print("Data loaded successfully:")
print(data.head())

# Function to check for convergence with adjustable parameters
def check_convergence(rewards, threshold, window):
    if len(rewards) < window:
        return False, 0
    recent_rewards = rewards[-window:]
    return np.all(np.abs(np.diff(recent_rewards)) < threshold), len(rewards)

# Function to calculate moving average
def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Function to calculate rolling variance
def rolling_variance(data, window_size):
    return data.rolling(window=window_size).var()

# Function to plot moving average rewards
def plot_moving_average(data, window_size):
    for agent in ["agent_0", "agent_1", "agent_2"]:
        data[f'{agent}_moving_avg'] = moving_average(data[f'avg_reward_{agent}'], window_size)

    plt.figure(figsize=(10, 6))
    for agent in ["agent_0", "agent_1", "agent_2"]:
        plt.plot(data['poison_probability'], data[f'{agent}_moving_avg'], label=f'{agent} Moving Avg')
    plt.xlabel('Disease Probability')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average Rewards vs Disease Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('collaboration_results/moving_avg_rewards.png')
    plt.show()

# Function to plot rolling variance of rewards
def plot_rolling_variance(data, window_size):
    for agent in ["agent_0", "agent_1", "agent_2"]:
        data[f'{agent}_rolling_var'] = rolling_variance(data[f'avg_reward_{agent}'], window_size)

    plt.figure(figsize=(10, 6))
    for agent in ["agent_0", "agent_1", "agent_2"]:
        plt.plot(data['poison_probability'], data[f'{agent}_rolling_var'], label=f'{agent} Rolling Var')
    plt.xlabel('Disease Probability')
    plt.ylabel('Rolling Variance')
    plt.title('Rolling Variance of Rewards vs Disease Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('collaboration_results/rolling_var_rewards.png')
    plt.show()

# Function to plot cumulative rewards
def plot_cumulative_rewards(data):
    data['cumulative_reward_agent_0'] = data['avg_reward_agent_0'].cumsum()
    data['cumulative_reward_agent_1'] = data['avg_reward_agent_1'].cumsum()
    data['cumulative_reward_agent_2'] = data['avg_reward_agent_2'].cumsum()

    plt.figure(figsize=(10, 6))
    for agent in ["agent_0", "agent_1", "agent_2"]:
        plt.plot(data['poison_probability'], data[f'cumulative_reward_{agent}'], label=f'{agent} Cumulative Reward')
    plt.xlabel('Disease Probability')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards vs Disease Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('collaboration_results/cumulative_rewards.png')
    plt.show()

# Function to analyze convergence for each agent with different parameters
def analyze_convergence(data):
    thresholds = [0.001, 0.005, 0.01, 0.02]
    window_sizes = [5, 10, 15, 20]
    epochs = list(range(10, 101))  # Check from 10 to 100 epochs

    convergence_log = []
    converged_flag = False
    for agent in ["agent_0", "agent_1", "agent_2"]:
        rewards = data[f'avg_reward_{agent}'].values
        for threshold in thresholds:
            for window in window_sizes:
                for epoch in epochs:
                    if epoch <= len(rewards):
                        converged, epochs_to_converge = check_convergence(rewards[:epoch], threshold, window)
                        convergence_log.append({
                            'agent': agent,
                            'threshold': threshold,
                            'window': window,
                            'epoch': epoch,
                            'converged': converged,
                            'epochs_to_converge': epochs_to_converge
                        })
                        if converged:
                            print(f"Agent {agent}: Converged = {converged}, Threshold = {threshold}, Window = {window}, Epochs = {epoch}")
                            converged_flag = True

    # Save the convergence log to a CSV file
    convergence_log_df = pd.DataFrame(convergence_log)
    convergence_log_df.to_csv('collaboration_results/convergence_log1.csv', index=False)

    return converged_flag

# Function to plot the final graph
def plot_final_graph(data):
    plt.figure(figsize=(10, 6))
    for agent in ["agent_0", "agent_1", "agent_2"]:
        plt.plot(data['poison_probability'], data[f'avg_reward_{agent}'], label=f'Agent {agent}')
    plt.xlabel('Disease Probability')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards vs Disease Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('collaboration_results/average_rewards2.png')
    plt.show()

# Main function to execute all the analyses and plots
def main():
    converged_flag = analyze_convergence(data)
    print(f"Converged Flag: {converged_flag}")

    plot_final_graph(data)

    if converged_flag:
        plot_moving_average(data, window_size=3)
        plot_rolling_variance(data, window_size=3)
        plot_cumulative_rewards(data)

if __name__ == "__main__":
    main()
