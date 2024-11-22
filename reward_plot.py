import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Load the CSV file
file_path = 'cloud_checkpoint/log.csv'  # Replace with your actual CSV file path
df = pd.read_csv(file_path)

# Define the window size for the sliding average
window_size = 10

# Calculate the sliding average for each agent
df['reward_agent_0_avg'] = df['agent_0'].rolling(window=window_size).mean()
df['reward_agent_1_avg'] = df['agent_1'].rolling(window=window_size).mean()
df['reward_agent_2_avg'] = df['agent_2'].rolling(window=window_size).mean()

# Env Config
config_path = 'cloud_checkpoint/config.yaml'
with open(config_path, 'r') as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)

# Divide the reward by number of turns
num_of_turns = env_config['env']['max_steps']
df['reward_agent_0_avg'] = df['reward_agent_0_avg'] / num_of_turns
df['reward_agent_1_avg'] = df['reward_agent_1_avg'] / num_of_turns
df['reward_agent_2_avg'] = df['reward_agent_2_avg'] / num_of_turns

# Plotting the smoothed rewards over time for each agent
plt.figure(figsize=(10, 6))
# plt.ylim(0, 1.5)

# Plotting each agent's smoothed reward
plt.plot(df.index, df['reward_agent_0_avg'], label='Agent 0', color='blue')
plt.plot(df.index, df['reward_agent_1_avg'], label='Agent 1', color='magenta')
plt.plot(df.index, df['reward_agent_2_avg'], label='Agent 2', color='yellow')


# Set the config display
config = str(env_config['env']['stag_poisoned_chance']) + str(env_config['env']['knows_poisoning'])
# config = env_config['env']['observation_mode']
# config = ""

# Adding title and labels
plt.title(f'[{config}] Avg Agent Reward Over Training Batches', fontsize=20)
plt.xlabel('Training Iteration', fontsize=16)
plt.ylabel('Reward per turn', fontsize=16)

# Adding a legend to differentiate between agents
plt.legend(fontsize=14)

# Display the plot
plt.show()

# Plot the second graph
eval_file_path = 'cloud_checkpoint/eval.csv'
df = pd.read_csv(eval_file_path)

plt.figure(figsize=(10, 6))
plt.ylim(0.7, 5)

plt.plot(df.index * 100, df['agent_0_mean_rewards'], label='Agent 0', color='blue')
plt.plot(df.index * 100, df['agent_1_mean_rewards'], label='Agent 1', color='magenta')
plt.plot(df.index * 100, df['agent_2_mean_rewards'], label='Agent 2', color='yellow')

plt.title(f'[{config}] Agent Reward Avg Over Evaluation Batches', fontsize=20)
plt.xlabel('Training Iteration', fontsize=16)
plt.ylabel('Reward Average', fontsize=16)

plt.legend(fontsize=14)

plt.show()

# Plot the third graph
plt.figure(figsize=(10, 6))
plt.ylim(0, 1)

plt.plot(df.index * 100, df['agent_0_stag_percentage'], label='Agent 0', color='blue')
plt.plot(df.index * 100, df['agent_1_stag_percentage'], label='Agent 1', color='magenta')
plt.plot(df.index * 100, df['agent_2_stag_percentage'], label='Agent 2', color='yellow')

plt.title(f'[{config}] Stag Percentage Over Evaluation Batches', fontsize=20)
plt.xlabel('Training Iteration', fontsize=16)
plt.ylabel('Stag Percentage', fontsize=16)

plt.legend(fontsize=14)

plt.show()