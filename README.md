# Multi Agent Reinforcement Learning in a Grid World Stag Hunt Game

MARL_GWSH is a one file RL environment that emphasizing flexibility, customizability, and ease of us. It supports heterogeneous agents with diverse observation and action spaces, enabling research and experimentation in multi-agent systems and complex scenarios. The environment is highly configurable through parameters and settings, allowing users to tailor it to their specific needs.  PyGame integration provides visualization capabilities for interactive monitoring and analysis.

# Installation

1. Make sure you are using Python 3.11
2. Install the required packages with `pip install -r requirements.txt`
3. (Optional) To speed up training, install CUDA

# The files

- The environment is [marl_gwsh.py](zoo_envs/env/marl_gwsh.py). It is a pettingzoo parallel environment. 
- [use_marl_gwsh.py](use_marl_gwsh.py) is an example of how to use the environment.
- [agent.py](agent.py) is the agent file that contains the base agent class, and a epsilon greedy agent class
- [hydra_rllib_training.py](hydra_rllib_training.py) is the file I use to train the agents with rllib and hydra
- [reward_plot.py](reward_plot.py) plots the rewards of the agents from the training in [hydra_rllib_training.py](hydra_rllib_training.py)
- [use_rllib_training.py](use_rllib_training.py) visualises playing the trained agents, config file is [use_config.yaml](conf/use_config.yaml)
- [config.yaml](conf/config.yaml) is the hydra configuration file for [hydra_rllib_training.py](hydra_rllib_training.py)

# How to use the environment

The most basic usage of the environment is as follows:

```python
from zoo_envs.env.marl_gwsh import MARL_GWSH
import numpy as np

env = MARL_GWSH()  # Create the environment
observations, infos = env.reset()  # Reset the environment and get the initial observations

while True:
    actions = {}  # Actions are a dictionary with the agent names as keys, and the actions as values
    actions["agent_0"] = np.random.randint(5)
    actions["agent_1"] = np.random.randint(5)
    actions["agent_2"] = np.random.randint(5)
    observations, rewards, terminations, truncations, infos = env.step(actions)  # Take a step in the environment

    # Here you can do whatever you want with the observations, rewards, terminations, and truncations
    env.render()  # (optional) Render the environment for visualisation
    if any(truncations.values()):
        break

env.close()
```

# Customizing the environment

The environment can be customised by passing arguments to `MARL_GWSH`, an alternative way would be to fill the `config.yaml` file with the desired parameters and use hydra to load the configuration file, like [use_marl_gwsh.py](use_marl_gwsh.py) does.

## Environment Configuration (`env`)

### Render Mode
- **render_mode**: Defines how the environment will be rendered.
  - `null`: No rendering is performed.
  - `console`: Text-based rendering in the console.
  - `human`: Graphical rendering using Pygame.

### Population Settings
- **number_of_agents**: Specifies the total number of agents in the environment (default: 3).
- **number_of_stags**: Specifies the total number of stags in the environment (default: 1).
- **number_of_plants**: Specifies the total number of plants in the environment (default: 1).

### Grid Dimensions
- **grid_width**: The width of the grid in which agents, stags, and plants operate (default: 5).
- **grid_height**: The height of the grid (default: 5).

### Rewards and Punishments
- **plant_reward**: Reward given to agents standing on a plant, divided by the number of agents on it (default: 1).
- **stag_reward**: Reward for each agent standing on a stag (default: 5).
- **mauling_punishment**: Punishment for agents standing on a stag without capturing it (default: 0).
- **capture_power_needed_for_stag**: Exact number of agents required to capture a stag (default: 2).
- **overcrowded_rids_stag**: If true, overcrowding on a stag results in its capture with no reward (default: true).
- **movement_punishment**: Punishment for moving (default: 0).

### Episode Settings
- **max_steps**: Maximum number of steps allowed in an episode before it is truncated (default: 150).

### Agent Configurations
- **agent_configs**: Specific configurations for each agent. If set to `null`, default values are used.
  - Each agent has:
    - **capture_power**: The power of the agent to capture stags (default: 1).
    - **movement**: Currently not utilized in the simulation (default: 1).

```yaml
  agent_configs:
    agent_0:
      capture_power: 1
      movement: 1
    agent_1:
      capture_power: 1
      movement: 1
    agent_2:
      capture_power: 1
      movement: 1      
```

### Reset Configurations
- **reset_configs**: Determines the initial positions of agents, stags, and plants when the environment resets. If set to `null`, positions are chosen randomly.
  - **stag_positions**: List of coordinates for stag(s) (default: [(0, 0)]).
  - **plant_positions**: List of coordinates for plant(s) (default: [(4, 4)]).
  - **agent_positions**: Dictionary defining the starting positions of each agent:
    - `agent_0`: (0, 4)
    - `agent_1`: (4, 0)
    - `agent_2`: (4, 4)

```yaml
 reset_configs:
    stag_positions: [(0, 0)]
    plant_positions: [(4, 4)]
    agent_positions: 
      agent_0: (0, 4)
      agent_1: (4, 0)
      agent_2: (4, 4)
```

### Observation and Action Modes

- **observation_mode**: Specifies how agents perceive their surroundings. The available modes are:

  - **one_hot**: 
    - Provides a one-hot encoding of the positions of agents, stags, and plants.
    - Includes one-hot encoding of the last action taken by the agent.
    - Also includes the rewards received by all agents.

  - **one_hot_2**: 
    - Similar to `one_hot`, but only includes the one-hot encoding of positions for agents, stags, and plants.
    - Does not include last actions or rewards.

  - **one_hot_3**: 
    - Similar to `one_hot_2`, but the agent's own position is always listed first.
    - This prioritization can help the agent focus on its own position relative to others.

  - **relative**: 
    - Provides the agent's position along with the positions of stags and plants relative to itself.
    - Relative positions are clipped, meaning negative values indicate positions that are on the opposite side of the grid.

  - **relative_2**: 
    - Similar to `relative`, but changes the order of observations.
    - Other agents’ positions are presented in numerical order, starting from the agent's number and wrapping around.
    - For example, if the agent is `agent_2` in a four-agent setup, the order would be `2, 3, 4, 1`.

  - **relative_3**: 
    - Provides unclipped and centered relative positions of other agents, stags, and plants.
    - This means that the positions maintain their spatial relationships, where what is to the right on the grid is also to the right in the observation.

  - **relative_4**: 
    - Similar to `relative_3`, but the other agent's relative position are arranged randomly.
  
- **action_mode**: Defines how actions are submitted by agents.
  - `single`: Each agent submits a single integer as an action.
  - `multi`: Each agent submits a list of integers. Used when agents are allowed to take multiple actions per turn.

### Respawn Settings
- **stag_respawn**: Indicates whether stags respawn after being captured (default: true).
- **plant_respawn**: Indicates whether plants respawn after being eaten (default: true).
- **reset_simplified**: If true, stags and plants only spawn on even coordinates (default: false).

### Stag Poisoning Mechanics
- **stag_poisoned_chance**: Probability that a stag is poisoned (default: 0).
- **knows_poisoning**: List of agents that are aware of whether a stag is poisoned (default: `null`).
  - If some agents know, the value would be `["agent_0", "agent_1"]`.
- **poison_death_turn**: Number of turns before a poisoned stag dies; `0` means it never dies (default: 0).
- **poison_capture_punishment**: Punishment for capturing a poisoned stag (default: 0).
- **limited_vision**: List of agents with limited vision (default: `null`).
  - If some agents have limited vision, the value would be `["agent_0", "agent_1"]`.

## Trying the environment

After you configured the environment in `use_config.ymal`, you can run the code in `use_marl_gwsh.py` to see the environment in action.

# Training agents with rllib

## Single training

Default training configuration is provided. The train the agent on various scenarios, we can use the `hydra_rllib_training.py` file.

There area several extra settings for training in the `config.yaml` file:

## Training Configuration Breakdown

### Training Configuration Breakdown

- **trial_count**: 
  - Default: `1`
  - It is intended for multi-run scenarios, allowing for performance evaluation across different trials.

- **num_iterations**: 
  - Default: `3000`
  - Specifies the total number of batches used to train the agent. Each batch consists of **512 episodes**, allowing agents to learn from a significant amount of experience. The duration of each episode is determined by the configurations set in the environment section.

- **checkpoint_interval**: 
  - Default: `500`
  - Defines how frequently checkpoints are saved during training. Checkpoints are useful for allowing the training process to be paused and resumed, enabling replay or analysis of past training sessions.

- **evaluation_interval**:
  - Default: `100`, if set to `0`, no evaluation is performed.
  - How often the capture rate of stag and plant is evaluated. Note that agent average reward will already be logged every training batch. This is to run the trained model in the environment and see how often it goes for stag and plant.

- **time_report_interval**: 
  - Default: `10`
  - Specifies how often the training time is reported. This helps in monitoring the progress of the training process and understanding how long it takes to complete batches.

- **checkpoints_dir**: 
  - Default: `checkpoints`
  - Indicates the directory where the checkpoints will be saved. This allows for organized storage of training states, making it easier to manage and access saved checkpoints.

For checkpoints_dir, the checkpoints directory will automatically increment if it already exists, so you can run multiple trainings without overwriting the previous checkpoints.

After everything is configured, we can run the training with `python hydra_rllib_training.py`.

## Multi-run training

The benefit of using hydra is it enables us to train in multiple settings with a single command. The following is an example:

`python hydra_rllib_training.py hydra.mode=MULTIRUN env.observation_mode=relative,relative_2`

This example trains the algorithm two times, the first one with all the settings in config.yaml but with env.observation_mode set to relative, the second one with env.observation_mode set to relative_2. Do not put space between the different options.

We can also try combinations of different settings, like:

`python hydra_rllib_training.py hydra.mode=MULTIRUN env.observation_mode=relative,relative_2 env.reset_simplified=true,false`

This will train the algorithm 4 times, with all the combinations of the two settings.

# Visualizing the training

After training, we can visualize the rewards of the agents with `python reward_plot.py`. This shows the reward of each agent during the training.

# Using the trained agents

To use the trained agents, paste the content of the trained agent's checkpoint directory to the `cloud_checkpoints` directory. There should be a `algorithm_state.pkl`, `rllib_checkpoint.json`, along with a `policies` folder, with all the agent folders in there. For each agent folder, there will be a `policy_state.pkl` and `rllib_checkpoint.json`.

Then you can run `python use_rllib_training.py`, remember to set the correct `env_config` in the code, and the training will be applied, and the agents will play out their action.