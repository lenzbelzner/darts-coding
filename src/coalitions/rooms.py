import gym
import numpy
import pathlib
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plot
import random
import copy
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from random import shuffle

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3
MOVE_NORTH_S = 4
MOVE_SOUTH_S = 5
MOVE_WEST_S = 6
MOVE_EAST_S = 7

ROOMS_ACTIONS = [MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST]
ROOMS_ACTIONS_S = [MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST,
                 MOVE_NORTH_S, MOVE_SOUTH_S, MOVE_WEST_S, MOVE_EAST_S]

AGENT_CHANNEL = 0
GOAL_CHANNEL = 1
OBSTACLE_CHANNEL = 2
OTHER_AGENTS_CHANNEL = 3
NR_CHANNELS = len([AGENT_CHANNEL, GOAL_CHANNEL,
                   OBSTACLE_CHANNEL])

NO_COMMITMENT = 0
ALL_OUT = 1
ANY_OUT = 2
COMMIT_MODES = [NO_COMMITMENT, ALL_OUT, ANY_OUT]

class RoomsEnv(gym.Env):

    def __init__(self, width, height, n_agents, obstacles, time_limit, stochastic=False, movie_filename=None, commit_mode=NO_COMMITMENT):
        self.seed()
        self.movie_filename = movie_filename
        self.commit_mode = commit_mode
        if self.commit_mode > 0:
            self.action_space = spaces.Discrete(len(ROOMS_ACTIONS_S))
        else:
            self.action_space = spaces.Discrete(len(ROOMS_ACTIONS))
        self.observation_space = spaces.Box(-numpy.inf,
                                            numpy.inf, shape=(NR_CHANNELS, width, height))
        self.n_agents = n_agents
        self.agent_positions = None
        self.ensemble = False
        self.done = False
        self.goal_position = (width-2, height-2)
        self.obstacles = obstacles
        self.time_limit = time_limit
        self.time = 0
        self.width = width
        self.height = height
        self.stochastic = stochastic
        self.undiscounted_returns = [0] * self.n_agents
        self.state_history = []
        self.reset()

    def state(self, i):
        state = numpy.zeros((NR_CHANNELS, self.width, self.height))
        
        for j, pos in enumerate(self.agent_positions):
            x, y = self.agent_positions[j]
            state[AGENT_CHANNEL][x][y] = 1

        x_goal, y_goal = self.goal_position
        state[GOAL_CHANNEL][x_goal][y_goal] = 1
        
        for obstacle in self.obstacles:
            x, y = obstacle
            state[OBSTACLE_CHANNEL][x][y] = 1
        
        return numpy.swapaxes(state, 0, 2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.stochastic:
            for i, a in enumerate(action):
                if numpy.random.rand() < 0.1:
                    action[i] = random.choice(ROOMS_ACTIONS)
        return self.step_with_action(action)

    def step_with_action(self, action):
        rewards = [0] * self.n_agents

        if self.done:
            return (self.agent_positions, self.ensemble), rewards, self.done, {}

        self.time += 1
        self.state_history.append(self.state(0))
        hit_obstacles = [False] * self.n_agents
        commits = [False] * self.n_agents
        
        actions_with_indices = [(i, a) for i, a in enumerate(action)]
        shuffle(actions_with_indices)
        for i, a in actions_with_indices:
            x, y = self.agent_positions[i]

            if a > 3:
                commits[i] = True
                a %= 4

            if a == MOVE_NORTH and y + 1 < self.height:
                new_position = (x, y + 1)
            elif a == MOVE_SOUTH and y - 1 >= 0:
                new_position = (x, y - 1)
            elif a == MOVE_WEST and x - 1 >= 0:
                new_position = (x - 1, y)
            elif a == MOVE_EAST and x + 1 < self.width:
                new_position = (x + 1, y)

            hit_obstacles[i] = self.set_position_if_no_obstacle(new_position, i)

            goal_reached = self.agent_positions[i] == self.goal_position
            if goal_reached:
                rewards[i] += (i + 1)
            self.undiscounted_returns[i] += rewards[i]
            self.done = self.done or goal_reached

        if all(commits):
            self.ensemble = True

        if self.commit_mode == 1:
            if all([not commit for commit in commits]):
                self.ensemble = False
        elif self.commit_mode == 2:
            if any([not commit for commit in commits]):
                self.ensemble = False

        if self.ensemble:
            rewards = [numpy.mean(rewards)] * self.n_agents

        self.done = self.done or self.time >= self.time_limit
        return (self.agent_positions, self.ensemble), rewards, self.done, {'hit_obstacle': hit_obstacles}

    def set_position_if_no_obstacle(self, new_position, i):
        if new_position not in self.obstacles and new_position not in self.agent_positions:
            self.agent_positions[i] = new_position
            return False
        return True  # hit obstacle

    def reset(self):
        self.done = False
        self.ensemble = False
        self.agent_positions = [(1, 1)] * self.n_agents
        self.time = 0
        self.state_history.clear()
        return self.agent_positions

    # TODO: used?
    def state_summary(self, state):
        # TODO: other agents
        return {
            "agent_x": self.agent_position[0],
            "agent_y": self.agent_position[0],
            "goal_x": self.goal_position[0],
            "goal_y": self.goal_position[0],
            "is_subgoal": self.is_subgoal(state),
            "time_step": self.time,
            "score": self.undiscounted_return
        }

    def save_video(self):
        if self.movie_filename is not None:
            history_of_states = self.state_history
            duration_in_seconds = len(history_of_states)/4
            fig, ax = plot.subplots()
            frames_per_second = len(history_of_states)/duration_in_seconds

            def make_frame(t):
                ax.clear()
                ax.grid(False)
                ax.imshow(history_of_states[int(t*frames_per_second)])
                ax.tick_params(axis='both', which='both', bottom=False, top=False,
                               left=False, right=False, labelleft=False, labelbottom=False)
                return mplfig_to_npimage(fig)
            animation = VideoClip(make_frame, duration=duration_in_seconds)
            animation.write_videofile(
                self.movie_filename, fps=frames_per_second)


def read_map_file(path):
    file = pathlib.Path(path)
    assert file.is_file()
    with open(path) as f:
        content = f.readlines()
    obstacles = []
    width = 0
    height = 0
    for y, line in enumerate(content):
        for x, cell in enumerate(line.strip().split()):
            if cell == '#':
                obstacles.append((x, y))
            width = x
        height = y
    width += 1
    height += 1
    return width, height, obstacles


def load_env(n_agents, path, movie_filename, time_limit=200, stochastic=False, commit_mode=NO_COMMITMENT):
    width, height, obstacles = read_map_file(path)
    return RoomsEnv(width, height, n_agents, obstacles, time_limit, stochastic, movie_filename, commit_mode)
