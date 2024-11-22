# This is a multi-agent reinforcement learning environment for the grid world stag hunt game.
# There are a customizable number of agents, stag, and plants
# the grid size is also customizable
import functools
import os
import random
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Dict, Tuple, Sequence, Box, MultiDiscrete
from .constants import *
import pygame


class MARL_GWSH(ParallelEnv):
    metadata = {"render_modes": ["none", "console", "human"],
                "observation_modes": ["coords", "flat_coords", "one_hot"],
                "action_modes": ["single", "multi"],
                "name": "marl_gwsh_v0"}

    def __init__(self, render_mode=None,
                 number_of_agents=2,
                 number_of_stags=1,
                 number_of_plants=1,
                 grid_width=5,
                 grid_height=5,
                 plant_reward=1,
                 stag_reward=5,
                 mauling_punishment=0,
                 capture_power_needed_for_stag=2,
                 max_steps=10,
                 agent_configs=None,
                 reset_configs=None,
                 observation_mode="coords",
                 action_mode="single",
                 stag_respawn=False,
                 plant_respawn=False,
                 reset_simplified=False,
                 stag_poisoned_chance=0,
                 knows_poisoning=None,
                 poison_death_turn=0,
                 poison_capture_punishment=0,
                 overcrowded_rids_stag=False,
                 movement_punishment=0,
                 limited_vision=None,
                 ):
        # Necessary attributes
        self.possible_agents = [f"agent_{i}" for i in range(number_of_agents)]
        self.render_mode = None if render_mode == "none" else render_mode

        # Agent attributes agent_configs = {"agent_0": {"capture_power": 1, "movement": 1}, "agent_1": {"capture_power":
        # 2, "movement": 2}, ...}
        if agent_configs is None:
            agent_configs = {agent: {"capture_power": 1, "movement": 1} for agent in self.possible_agents}
        self.agent_configs = agent_configs

        # Configs for the game
        self.reset_configs = reset_configs
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.stag_respawn = stag_respawn
        self.plant_respawn = plant_respawn
        self.reset_simplified = reset_simplified

        self.overcrowded_rids_stag = overcrowded_rids_stag
        self.movement_punishment = movement_punishment

        # Game attributes
        self.step_count = None
        self.grid_size = (grid_width, grid_height)
        self.number_of_stags = number_of_stags
        self.number_of_plants = number_of_plants
        self.plant_reward = plant_reward
        self.stag_reward = stag_reward
        self.mauling_punishment = mauling_punishment
        self.capture_power_needed_for_stag = capture_power_needed_for_stag
        self.max_steps = max_steps
        self.stag_positions = []
        self.plant_positions = []
        self.agent_positions = {}
        self.rewards_for_one_hot_encoding = {agent: 0 for agent in self.possible_agents}
        self.action_for_one_hot_encoding = {agent: 0 for agent in self.possible_agents}

        # Stag poisoned logic
        self.stag_poisoned_chance = stag_poisoned_chance
        self.stag_poisoned = set() # positions
        self.knows_poisoning = [] if knows_poisoning is None else knows_poisoning
        self.poison_death_turn = poison_death_turn
        self.poison_death_countdown = {} # position to int
        self.poison_capture_punishment = poison_capture_punishment
        self.limited_vision = [] if limited_vision is None else limited_vision

        # Pygame stuff if human rendering is used
        if self.render_mode == "human":
            x = 600
            y = 10
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)  # determines where the window spawns
            pygame.init()
            pygame.display.set_mode((1, 1), pygame.NOFRAME)
            self._screen_size = SCREEN_SIZE
            self._screen = pygame.display.set_mode(self._screen_size)
            self._clock = pygame.time.Clock()
            game_surface_size = (self.grid_size[0] * TILE_SIZE, self.grid_size[1] * TILE_SIZE)
            self._background = pygame.Surface(game_surface_size).convert()
            self._background.fill(BACKGROUND_COLOR)
            self._grid_layer = pygame.Surface(game_surface_size).convert_alpha()
            self._grid_layer.fill(CLEAR)
            self._entity_layer = pygame.Surface(game_surface_size).convert_alpha()
            self._entity_layer.fill(CLEAR)
            self._draw_grid()
            pygame.display.flip()

    def reset(self, seed=None, options=None):
        if options is None:
            if self.reset_configs is not None:
                options = self.reset_configs
            else:
                # options = {
                #     'stag_positions': [(0, 0)],
                #     'plant_positions': [(4, 4)],
                #     'agent_positions': {'agent_0': (0, 4), 'agent_1': (4, 0)}
                # }
                options = {}
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.stag_positions = []
        self.plant_positions = []
        self.agent_positions = {}

        # Place entities according to options if specified, else randomly
        valid_coordinates = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        if self.reset_simplified:
            valid_coordinates = [(x, y) for x, y in valid_coordinates if x % 2 == 0 and y % 2 == 0]
        random.shuffle(valid_coordinates)

        if "stag_positions" in options:
            self.stag_positions = options["stag_positions"].copy()
        else:
            for i in range(self.number_of_stags):
                self.stag_positions.append(valid_coordinates.pop())

        if "plant_positions" in options:
            self.plant_positions = options["plant_positions"].copy()
        else:
            for i in range(self.number_of_plants):
                self.plant_positions.append(valid_coordinates.pop())

        if "agent_positions" in options:
            self.agent_positions = options["agent_positions"].copy()
        else:
            for agent in self.possible_agents:
                self.agent_positions[agent] = valid_coordinates.pop()

        self._poison_stag_logic(self.stag_positions)

        observations = self.get_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        # actions = {agent_0: [0], agent_1: [0,1], ...}
        self.step_count += 1

        # Process collision logic and rewards
        rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Move agents
        for agent in self.agents:
            position = self.agent_positions[agent]
            individual_actions = actions[agent]
            if self.action_mode == "single":
                individual_actions = [individual_actions]
            new_position = self._move(position, individual_actions)
            # give movement punishment
            for action in individual_actions:
                if action != 0:
                    rewards[agent] += self.movement_punishment
            self.agent_positions[agent] = new_position

        # Collision with plants
        for plant_pos in self.plant_positions[:]:
            agents_on_plant = [agent for agent in self.agents if self.agent_positions[agent] == plant_pos]
            for agent in agents_on_plant:
                # PLANT REWARD
                # the reward is divided among all agents on the plant
                rewards[agent] += self.plant_reward / len(agents_on_plant)
                infos[agent]["capture"] = "plant"
            if len(agents_on_plant) > 0:
                self.plant_positions.remove(plant_pos)
                if self.plant_respawn:
                    new_plant_pos = self._find_random_valid_coord()
                    self.plant_positions.append(new_plant_pos)

        # Collision with stag
        for stag_pos in self.stag_positions[:]:
            # a certain amount of capture power is needed to capture the stag
            total_capture_power = 0
            agents_on_stag = []
            for agent in self.agents:
                if self.agent_positions[agent] == stag_pos:
                    total_capture_power += self.agent_configs[agent]["capture_power"]
                    agents_on_stag.append(agent)
            # POISONED STAG CAPTURE LOGIC
            if stag_pos in self.stag_poisoned and total_capture_power > 0:
                # can be captured by any capture power
                for agent in agents_on_stag:
                    rewards[agent] += self.poison_capture_punishment
                    infos[agent]["capture"] = "poisoned_stag"
                self.stag_positions.remove(stag_pos)
                self.stag_poisoned.remove(stag_pos)
                self.poison_death_countdown.pop(stag_pos)
                if self.stag_respawn:
                    new_stag_pos = self._find_random_valid_coord()
                    self.stag_positions.append(new_stag_pos)
                    self._poison_stag_logic([new_stag_pos])

            # STAG CAPTURE LOGIC
            elif total_capture_power == self.capture_power_needed_for_stag:
                # only when capture power is equal to the required amount reward is given
                for agent in agents_on_stag:
                    rewards[agent] += self.stag_reward
                    infos[agent]["capture"] = "stag"
                self.stag_positions.remove(stag_pos)
                if self.stag_respawn:
                    new_stag_pos = self._find_random_valid_coord()
                    self.stag_positions.append(new_stag_pos)
                    self._poison_stag_logic([new_stag_pos])

            elif total_capture_power > self.capture_power_needed_for_stag:
                # if the total capture power is more than required, no reward is given
                if self.overcrowded_rids_stag:
                    self.stag_positions.remove(stag_pos)
                    if self.stag_respawn:
                        new_stag_pos = self._find_random_valid_coord()
                        self.stag_positions.append(new_stag_pos)
                        self._poison_stag_logic([new_stag_pos])
            else:
                # if the total capture power is less than required, punishment is given
                for agent in agents_on_stag:
                    rewards[agent] += self.mauling_punishment

        terminations = {agent: False for agent in self.agents}

        # Check if the game is over
        env_truncation = self.step_count >= self.max_steps
        truncations = {agent: env_truncation for agent in self.agents}

        # stag died of poisoning logic
        for stag_pos in self.stag_positions[:]:
            if stag_pos in self.stag_poisoned and self.poison_death_turn > 0:
                self.poison_death_countdown[stag_pos] -= 1
                if self.poison_death_countdown[stag_pos] < 0:
                    self.stag_positions.remove(stag_pos)
                    self.stag_poisoned.remove(stag_pos)
                    self.poison_death_countdown.pop(stag_pos)
                    if self.stag_respawn:
                        new_stag_pos = self._find_random_valid_coord()
                        self.stag_positions.append(new_stag_pos)
                        self._poison_stag_logic([new_stag_pos])

        if self.observation_mode == "one_hot":
            for agent in self.agents:
                self.rewards_for_one_hot_encoding[agent] = rewards[agent]
                self.action_for_one_hot_encoding[agent] = actions[agent]

        observations = self.get_observations()

        self.render()

        return observations, rewards, terminations, truncations, infos

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        num_agents = len(self.possible_agents)
        if self.observation_mode == "coords":
            return Dict({
                "stag_positions": Sequence(Tuple((Discrete(self.grid_size[0]), Discrete(self.grid_size[1])))),
                "plant_positions": Sequence(Tuple((Discrete(self.grid_size[0]), Discrete(self.grid_size[1])))),
                "agent_positions": Dict({
                    agent: Tuple((Discrete(self.grid_size[0]), Discrete(self.grid_size[1])))
                    for agent in self.possible_agents
                })
            })
        elif self.observation_mode == "flat_coords":
            # a box of the position of stag, plant, and all agents, with -1 for stags and plants that doesn't exist anymore
            num_entities = self.number_of_stags + self.number_of_plants + num_agents
            return Box(low=0, high=max(self.grid_size), shape=(num_entities, 2), dtype=np.int32)
        elif self.observation_mode == "one_hot":
            # one hot encoding of each of the item in the grid, then rewards, and the most recent agent action
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents)
            num_cells = (self.grid_size[0] * self.grid_size[1])
            return Box(-1.0, 1.0, (num_entities * num_cells + num_agents + 5,), dtype=np.float32)
        elif self.observation_mode == "one_hot_2":
            # one hot encoding of each of the item in the grid
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents)
            num_cells = (self.grid_size[0] * self.grid_size[1])
            return Box(-1.0, 1.0, (num_entities * num_cells,), dtype=np.float32)
        elif self.observation_mode == "one_hot_3":
            # one hot encoding of each of the item in the grid, it's position always comes first
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents)
            num_cells = (self.grid_size[0] * self.grid_size[1])
            return Box(-1.0, 1.0, (num_entities * num_cells,), dtype=np.float32)
        elif self.observation_mode == "relative":
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents)
            num_cells = (self.grid_size[0] * self.grid_size[1])
            return Box(-1.0, 1.0, (num_entities * num_cells,), dtype=np.float32)
        elif self.observation_mode == "relative_2":
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents)
            num_cells = (self.grid_size[0] * self.grid_size[1])
            return Box(-1.0, 1.0, (num_entities * num_cells,), dtype=np.float32)
        elif self.observation_mode == "relative_3":
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents) - 1
            width = (self.grid_size[0] - 1) * 2 + 1
            height = (self.grid_size[1] - 1) * 2 + 1
            num_cells = width * height
            return Box(-1.0, 1.0, (num_entities * num_cells,), dtype=np.float32)
        elif self.observation_mode == "relative_4":
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents) - 1
            width = (self.grid_size[0] - 1) * 2 + 1
            height = (self.grid_size[1] - 1) * 2 + 1
            num_cells = width * height
            return Box(-1.0, 1.0, (num_entities * num_cells,), dtype=np.float32)
        elif self.observation_mode == "poison":
            # relative + poison status
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents)
            num_cells = (self.grid_size[0] * self.grid_size[1])
            return Box(-1.0, 1.0, (num_entities * num_cells + 1,), dtype=np.float32)
        elif self.observation_mode == "poison_2":
            # relative_3 + poison status
            num_entities = (self.number_of_stags + self.number_of_plants + num_agents) - 1
            width = (self.grid_size[0] - 1) * 2 + 1
            height = (self.grid_size[1] - 1) * 2 + 1
            num_cells = width * height
            return Box(-1.0, 1.0, (num_entities * num_cells + 1,), dtype=np.float32)

    def get_observations(self):
        if self.observation_mode == "coords":
            return {
                "stag_positions": self.stag_positions.copy(),
                "plant_positions": self.plant_positions.copy(),
                "agent_positions": self.agent_positions.copy()
            }
        elif self.observation_mode == "flat_coords":
            # a box of the position of stag, plant, and all agents, with -1 for stags and plants that doesn't exist anymore
            num_of_entities = self.number_of_stags + self.number_of_plants + len(self.possible_agents)
            observation = np.zeros((num_of_entities, 2), dtype=np.int32)
            for i, stag_pos in enumerate(self.stag_positions):
                observation[i] = stag_pos
            for i, plant_pos in enumerate(self.plant_positions):
                observation[self.number_of_stags + i] = plant_pos
            for i, agent in enumerate(self.possible_agents):
                observation[self.number_of_stags + self.number_of_plants + i] = self.agent_positions[agent]
            return {agent: observation for agent in self.possible_agents}
        elif self.observation_mode == "one_hot":
            rewards_normalized = [reward / self.stag_reward for reward in self.rewards_for_one_hot_encoding.values()]
            state = np.concatenate(
                [self._get_one_hot_matrix(self.agent_positions[agent]) for agent in self.possible_agents] +
                [self._get_one_hot_matrix(stag_pos) for stag_pos in self.stag_positions] +
                [self._get_one_hot_matrix(plant_pos) for plant_pos in self.plant_positions]
            )
            return {agent: np.concatenate(
                [state, rewards_normalized, self._get_one_hot_action(self.action_for_one_hot_encoding[agent])]) for
                agent in self.possible_agents}
        elif self.observation_mode == "one_hot_2":
            state = np.concatenate(
                [self._get_one_hot_matrix(self.agent_positions[agent]) for agent in self.possible_agents] +
                [self._get_one_hot_matrix(stag_pos) for stag_pos in self.stag_positions] +
                [self._get_one_hot_matrix(plant_pos) for plant_pos in self.plant_positions]
            )
            return {agent: state for agent in self.possible_agents}
        elif self.observation_mode == "one_hot_3":
            agents_obs = {}
            for agent in self.possible_agents:
                # agent's position comes first
                state = np.concatenate(
                    [self._get_one_hot_matrix(self.agent_positions[agent])] +
                    # the remaining two agents' positions, in numerical order
                    [self._get_one_hot_matrix(self.agent_positions[other_agent]) for other_agent in self.possible_agents
                     if other_agent != agent] +
                    [self._get_one_hot_matrix(stag_pos) for stag_pos in self.stag_positions] +
                    [self._get_one_hot_matrix(plant_pos) for plant_pos in self.plant_positions]
                )
                agents_obs[agent] = state
            return agents_obs
        elif self.observation_mode == "relative":
            agents_obs = {}
            for agent in self.possible_agents:
                # calculate the relative position of all the entities relative to the agent
                state = np.concatenate(
                    [self._get_one_hot_matrix(self.agent_positions[agent])] +
                    # the remaining two agents' positions, in numerical order
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent],
                                                                          self.agent_positions[other_agent])) for
                     other_agent in self.possible_agents if other_agent != agent] +
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent], stag_pos)) for
                     stag_pos in self.stag_positions] +
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent], plant_pos)) for
                     plant_pos in self.plant_positions]
                )
                agents_obs[agent] = state
            return agents_obs
        elif self.observation_mode == "relative_2":
            agents_obs = {}
            for agent_index, agent in enumerate(self.possible_agents):
                # calculate the relative position of all the entities relative to the agent
                other_agent_positions = []
                for i in range(len(self.possible_agents) - 1):
                    index = (i + agent_index + 1) % len(self.possible_agents)
                    other_agent_positions.append(self.agent_positions[self.possible_agents[index]])
                state = np.concatenate(
                    [self._get_one_hot_matrix(self.agent_positions[agent])] +
                    # the remaining two agents' positions
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent], pos)) for pos in
                     other_agent_positions] +
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent], stag_pos)) for
                     stag_pos in self.stag_positions] +
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent], plant_pos)) for
                     plant_pos in self.plant_positions]
                )
                agents_obs[agent] = state
            return agents_obs
        elif self.observation_mode == "relative_3":
            agents_obs = {}
            for agent_index, agent in enumerate(self.possible_agents):
                # calculate the relative position of all the entities relative to the agent
                other_agent_positions = []
                for i in range(len(self.possible_agents) - 1):
                    index = (i + agent_index + 1) % len(self.possible_agents)
                    other_agent_positions.append(self.agent_positions[self.possible_agents[index]])
                state = np.concatenate(
                    # the remaining two agents' positions
                    [self._get_one_hot_matrix_middle_centered(
                        self._get_relative_position(self.agent_positions[agent], pos)) for pos in
                        other_agent_positions] +
                    [self._get_one_hot_matrix_middle_centered(
                        self._get_relative_position(self.agent_positions[agent], stag_pos)) for
                        stag_pos in self.stag_positions] +
                    [self._get_one_hot_matrix_middle_centered(
                        self._get_relative_position(self.agent_positions[agent], plant_pos)) for
                        plant_pos in self.plant_positions]
                )
                agents_obs[agent] = state
            return agents_obs
        elif self.observation_mode == "relative_4":
            agents_obs = {}
            for agent_index, agent in enumerate(self.possible_agents):
                # calculate the relative position of all the entities relative to the agent
                other_agent_positions = []
                for i in range(len(self.possible_agents) - 1):
                    index = (i + agent_index + 1) % len(self.possible_agents)
                    other_agent_positions.append(self.agent_positions[self.possible_agents[index]])
                np.random.shuffle(other_agent_positions)
                state = np.concatenate(
                    # the remaining two agents' positions
                    [self._get_one_hot_matrix_middle_centered(
                        self._get_relative_position(self.agent_positions[agent], pos)) for pos in
                        other_agent_positions] +
                    [self._get_one_hot_matrix_middle_centered(
                        self._get_relative_position(self.agent_positions[agent], stag_pos)) for
                        stag_pos in self.stag_positions] +
                    [self._get_one_hot_matrix_middle_centered(
                        self._get_relative_position(self.agent_positions[agent], plant_pos)) for
                        plant_pos in self.plant_positions]
                )
                agents_obs[agent] = state
            return agents_obs
        elif self.observation_mode == "poison":
            agents_obs = {}
            for agent in self.possible_agents:
                # calculate the relative position of all the entities relative to the agent
                state = np.concatenate(
                    [self._get_one_hot_matrix(self.agent_positions[agent])] +
                    # the remaining two agents' positions, in numerical order
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent],
                                                                          self.agent_positions[other_agent])) for
                     other_agent in self.possible_agents if other_agent != agent] +
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent], stag_pos)) for
                     stag_pos in self.stag_positions] +
                    [self._get_one_hot_matrix(self._get_relative_position(self.agent_positions[agent], plant_pos)) for
                     plant_pos in self.plant_positions] +
                    [np.array([1 if self.stag_poisoned else 0]) if agent in self.knows_poisoning else np.array([0])]
                )
                agents_obs[agent] = state
            return agents_obs
        elif self.observation_mode == "poison_2":
            agents_obs = {}
            for agent_index, agent in enumerate(self.possible_agents):
                # calculate the relative position of all the entities relative to the agent
                other_agent_positions = []
                for i in range(len(self.possible_agents) - 1):
                    index = (i + agent_index + 1) % len(self.possible_agents)
                    other_agent_positions.append(self.agent_positions[self.possible_agents[index]])
                if agent in self.limited_vision:
                    state = np.concatenate(
                        # the remaining two agents' positions
                        [self._get_one_hot_matrix_middle_centered_limited_vision(
                            self._get_relative_position(self.agent_positions[agent], pos)) for pos in
                            other_agent_positions] +
                        [self._get_one_hot_matrix_middle_centered_limited_vision(
                            self._get_relative_position(self.agent_positions[agent], stag_pos)) for
                            stag_pos in self.stag_positions] +
                        [self._get_one_hot_matrix_middle_centered_limited_vision(
                            self._get_relative_position(self.agent_positions[agent], plant_pos)) for
                            plant_pos in self.plant_positions] +
                        [np.array([1 if (self.stag_poisoned and (agent in self.knows_poisoning)) else 0])]
                    )
                else:
                    state = np.concatenate(
                        # the remaining two agents' positions
                        [self._get_one_hot_matrix_middle_centered(
                            self._get_relative_position(self.agent_positions[agent], pos)) for pos in
                            other_agent_positions] +
                        [self._get_one_hot_matrix_middle_centered(
                            self._get_relative_position(self.agent_positions[agent], stag_pos)) for
                            stag_pos in self.stag_positions] +
                        [self._get_one_hot_matrix_middle_centered(
                            self._get_relative_position(self.agent_positions[agent], plant_pos)) for
                            plant_pos in self.plant_positions] +
                        [np.array([1 if (self.stag_poisoned and (agent in self.knows_poisoning)) else 0])]
                    )
                agents_obs[agent] = state
            return agents_obs

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.action_mode == "single":
            return Discrete(5)
        elif self.action_mode == "multi":
            return Sequence(Discrete(5))
        # return MultiDiscrete(np.array([5]))
        # a number of up, down, left, right or stand
        # return Sequence(Discrete(5))

    def render(self):
        if self.render_mode == None:
            return

        if self.render_mode == "console":
            grid = np.full((self.grid_size[0], self.grid_size[1]), ".",
                           dtype=np.dtype(f"U{len(self.possible_agents) + 1}"))
            for position in self.stag_positions:
                self._set_cell(grid, position, "S")
            for position in self.plant_positions:
                self._set_cell(grid, position, "P")
            for agent in self.possible_agents:
                self._set_cell(grid, self.agent_positions[agent], agent[6])
            print(grid)
            if self.stag_poisoned_chance > 0:
                print(f"Stag Poisoned: {self.stag_poisoned}")

        if self.render_mode == "human":
            pygame.event.get()
            self._background.fill(BACKGROUND_COLOR)
            self._background.blit(self._grid_layer, (0, 0))
            self._entity_layer.fill(CLEAR)
            for stag_pos in self.stag_positions:
                if stag_pos in self.stag_poisoned:
                    self._entity_layer.blit(self._load_image(POISONED_STAG_SPRITE),
                                            (stag_pos[0] * TILE_SIZE, stag_pos[1] * TILE_SIZE))
                else:
                    self._entity_layer.blit(self._load_image(STAG_SPRITE),
                                            (stag_pos[0] * TILE_SIZE, stag_pos[1] * TILE_SIZE))
            for plant_pos in self.plant_positions:
                self._entity_layer.blit(self._load_image(HARE_SPRITE),
                                        (plant_pos[0] * TILE_SIZE, plant_pos[1] * TILE_SIZE))
            size_offset = 0
            size_increase = -2
            position_offset = 0
            position_increase = 6 // len(self.agent_positions)
            for agent, pos in self.agent_positions.items():
                # pygame.draw.rect(self._entity_layer, AGENT_COLORS[agent], (
                #     pos[0] * TILE_SIZE + TILE_SIZE // 4, pos[1] * TILE_SIZE + TILE_SIZE // 4,
                #     TILE_SIZE // 2 + size_offset, TILE_SIZE // 2 + size_offset))
                agent_image = self._load_image(SPRITE_DICT[agent])
                # agent_image = pygame.transform.scale(agent_image, (TILE_SIZE + size_offset, TILE_SIZE + size_offset))
                self._entity_layer.blit(agent_image,
                                        (pos[0] * TILE_SIZE + position_offset, pos[1] * TILE_SIZE))
                # size_offset += size_increase
                position_offset += position_increase
            self._background.blit(self._entity_layer, (0, 0))
            surf = pygame.transform.scale(self._background, self._screen_size)
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()

    def _set_cell(self, grid, position, value):
        # Helper function to set the cell value
        if grid[(position[1], position[0])] == ".":
            grid[(position[1], position[0])] = value
        else:
            grid[(position[1], position[0])] = str(grid[(position[1], position[0])]) + str(value)

    def _get_one_hot_action(self, action):
        one_hot_action = np.zeros(5)
        one_hot_action[action] = 1
        return one_hot_action

    def _get_one_hot_matrix(self, coords):
        # Helper function to get the one hot matrix
        one_hot_matrix = np.zeros((self.grid_size[0], self.grid_size[1]))
        one_hot_matrix[coords[0], coords[1]] = 1
        return one_hot_matrix.flatten()

    def _get_one_hot_matrix_middle_centered(self, coords):
        # One hot matrix centered around the middle
        width = (self.grid_size[0] - 1) * 2 + 1
        height = (self.grid_size[1] - 1) * 2 + 1
        one_hot_matrix = np.zeros((width, height))
        one_hot_matrix[coords[0] + self.grid_size[0] - 1, coords[1] + self.grid_size[1] - 1] = 1
        return one_hot_matrix.flatten()

    def _get_one_hot_matrix_middle_centered_limited_vision(self, coords):
        # Only encode the position if within 2 squares of the agent
        width = (self.grid_size[0] - 1) * 2 + 1
        height = (self.grid_size[1] - 1) * 2 + 1
        one_hot_matrix = np.zeros((width, height))
        if abs(coords[0]) <= 2 and abs(coords[1]) <= 2:
            one_hot_matrix[coords[0] + self.grid_size[0] - 1, coords[1] + self.grid_size[1] - 1] = 1
        return one_hot_matrix.flatten()

    def _get_relative_position(self, agent_pos, other_pos):
        # Helper function to get the relative position
        return np.array([other_pos[0] - agent_pos[0], other_pos[1] - agent_pos[1]])

    def _move(self, position, actions):
        for action in actions:
            if action == UP:
                position = (position[0], max(0, position[1] - 1))
            elif action == DOWN:
                position = (position[0], min(self.grid_size[1] - 1, position[1] + 1))
            elif action == LEFT:
                position = (max(0, position[0] - 1), position[1])
            elif action == RIGHT:
                position = (min(self.grid_size[0] - 1, position[0] + 1), position[1])
        return position

    def _draw_grid(self):
        for x in range(0, self.grid_size[0] * TILE_SIZE, TILE_SIZE):
            pygame.draw.line(self._grid_layer, GRID_LINE_COLOR, (x, 0), (x, self.grid_size[1] * TILE_SIZE))
        for y in range(0, self.grid_size[1] * TILE_SIZE, TILE_SIZE):
            pygame.draw.line(self._grid_layer, GRID_LINE_COLOR, (0, y), (self.grid_size[0] * TILE_SIZE, y))

    def _load_image(self, file):
        return pygame.image.load(file).convert_alpha()

    def _find_random_valid_coord(self):
        # valid means within the grid and doesn't overlap with other entities
        # try 100 times to find a valid coordinate by random
        for _ in range(100):
            x = random.randint(0, self.grid_size[0] - 1)
            y = random.randint(0, self.grid_size[1] - 1)
            if (x, y) not in self.stag_positions and (x, y) not in self.plant_positions and \
                    all((x, y) != pos for pos in self.agent_positions.values()):
                return x, y
        # else do a linear search
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x, y) not in self.stag_positions and (x, y) not in self.plant_positions and \
                        all((x, y) != pos for pos in self.agent_positions.values()):
                    return x, y

    def _poison_stag_logic(self, stag_pos_to_poison):
        # Stag poisoning logic
        for stag_pos in stag_pos_to_poison:
            if np.random.random() < self.stag_poisoned_chance:
                self.stag_poisoned.add(stag_pos)
                self.poison_death_countdown[stag_pos] = self.poison_death_turn




def encode_env(observation):
    #         return {
    #             "stag_positions": self.stag_positions,
    #             "plant_positions": self.plant_positions,
    #             "agent_positions": self.agent_positions
    #         }
    # turn the observation into a tuple for hashing
    return (tuple(observation["stag_positions"]),
            tuple(observation["plant_positions"]),
            tuple([pos for pos in observation["agent_positions"].values()])
            )
