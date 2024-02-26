from contextlib import closing
from io import StringIO
from os import path
import random
from typing import List, Optional
import time

import numpy as np

import gym
from gym import Env, spaces, utils
from gym.error import DependencyNotInstalled
from typing import List, Tuple


class DragonBallEnv(Env):
    """
    dragon ball environment involves crossing a namik planet from Start(S) to Goal(G) without falling into any Holes(H)
    by walking over the planet.
    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:
    - 0: DOWN
    - 1: RIGHT
    - 2: UP
    - 3: LEFT
    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.
    ### Arguments
    `desc`: Used to specify custom map for namik planet. For example,
        desc=["SFFF", "FHFH", "FFFH", "HFFG"].
    """

    def __init__(
            self,
            desc
    ):
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.d1 = None
        self.d2 = None
        self.goals = []
        self.collected_dragon_balls = [False, False]  # to keep track of collected Dragon Balls

        nA = 4
        nL_cost = {b"F": 10.0, b"H": np.inf, b"T": 3.0, b"A": 2.0, b"L": 1.0, b"S": 1.0, b"G": 1.0, b"D": 1}
        nS = nrow * ncol

        self.dic = {(s, False, False): {a: [] for a in range(nA)} for s in range(nS)}
        self.P = {s[0]: {a: [] for a in range(nA)} for s in self.dic}

        for row in range(nrow):
            for col in range(ncol):
                state = self.to_state(row, col)
                if desc[row, col] == b"D":
                    if self.d1 == None:
                        self.d1 = state
                    else:
                        self.d2 = state
                if desc[row, col] == b"G":
                    state = (state[0], True, True)
                    self.goals.append(state)

        for row in range(nrow):
            for col in range(ncol):
                for action in range(nA):
                    new_row, new_col = self.inc(row, col, action)
                    state = self.to_state(row, col)
                    curlleter = desc[row, col]
                    newletter = desc[new_row, new_col]
                    newstate = self.to_state(new_row, new_col)
                    if newletter == b"D":
                        newstate = self.to_state(new_row, new_col)
                    if curlleter == b"H":
                        self.P[state[0]][action] = (None, None, None)
                    else:
                        terminated = bytes(newletter) in b"GH"
                        cost = nL_cost[newletter]
                        self.P[state[0]][action] = (newstate, cost, terminated)

        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(nS),
            spaces.Discrete(2),  # Boolean: Collected Dragon Ball 1
            spaces.Discrete(2)  # Boolean: Collected Dragon Ball 2
        ))

        self.render_mode = "ansi"


    def step(self, a: int) -> Tuple[Tuple, int, bool]:
        """
        Moving the agent one step.

        Args:
            a - action(DOWN, RIGHT, UP, LEFT)
        Returns:
            the new state, the cost of the step and whether the search is over 
            (it can happen when the agent reaches a final state or falls into a hole).
        """
        newstate, cost, terminated = self.P[self.s[0]][a]
        if newstate[0] == self.d1[0]:
            self.collected_dragon_balls[0] = True
        elif newstate[0] == self.d2[0]:
            self.collected_dragon_balls[1] = True

        newstate = (newstate[0], self.collected_dragon_balls[0], self.collected_dragon_balls[1])


        self.s = newstate
        self.lastaction = a

        return (newstate, cost, terminated)


    def inc(self, row: int, col: int, a: int) -> Tuple[int, int]:
        """
        Given a position and an action, returns the new position.

        Args:
            row - row
            col - col
            a - action
        Returns:
            The new position.
        """

        if a == 0:
            row = min(row + 1, self.nrow - 1)
        elif a == 1:
            col = min(col + 1, self.ncol - 1)
        elif a == 2:
            row = max(row - 1, 0)
        elif a == 3:
            col = max(col - 1, 0)
        return (row, col)

    def to_state(self, row: int, col: int) -> Tuple:
        """
        Converts between location on the board and state.
        Args:
            row
            col
        Returns:
            state
        """
        return (row * self.ncol + col, self.collected_dragon_balls[0], self.collected_dragon_balls[1])

    def to_row_col(self, state: Tuple) -> Tuple[int, int]:
        """
        Converts between state and location on the board.
        Args:
            state
        Returns:
            row, col
        """
        return (state[0] // self.ncol, state[0] % self.ncol)

    def succ(self, state: Tuple):
        """
        Returns the successors of the state.
        Args:
            state
        Returns:
            Returns a dictionary that contains information on all the successors of the state.
            The keys are the actions. 
            The values are tuples of the form (new state, cost, terminated). 
            Note that terminated is true when the agent reaches a final state or a hole.
        """
        return self.P[state[0]]

    def set_state(self, state: Tuple) -> None:
        """
        Sets the current state of the agent.
        """
        self.s = state
        if self.collected_dragon_balls[0] == False:
            self.collected_dragon_balls[0] = state[1]
        if self.collected_dragon_balls[1] == False:
            self.collected_dragon_balls[1] = state[2]


    def get_state(self):
        """
        Returns the current state of the agent.
        """
        return (self.s[0], self.collected_dragon_balls[0], self.collected_dragon_balls[1])

    def is_final_state(self, state: Tuple) -> bool:
        """
        Returns True if the state is a final state.
        The function can help you understand whether you have fallen
        into a hole or whether you have reached a final state
        """
        return state in self.goals

    def get_initial_state(self) -> Tuple:
        """
        Returns the initial state.
        """
        return 0, False, False

    def get_goal_states(self) -> List[Tuple]:
        return self.goals

    def reset(self) -> int:
        """
        Initializes the search problem. 
        """
        super().reset()
        self.s = self.get_initial_state()
        self.lastaction = None
        self.collected_dragon_balls = [False, False]
        return self.s

    def render(self):
        """
        Returns a view of the board. 
        """
        return self._render_text()

    def _render_text(self):
        desc = self.desc.copy().tolist()

        outfile = StringIO()

        row, col = self.s[0] // self.ncol, self.s[0] % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        for r in range(len(desc)):
            for c in range(len(desc[r])):
                if desc[r][c] == "H":
                    desc[r][c] = utils.colorize(desc[r][c], "white", highlight=True)
                elif desc[r][c] == "D":
                    desc[r][c] = utils.colorize(desc[r][c], "yellow", highlight=True)
                elif desc[r][c] == "G":
                    desc[r][c] = utils.colorize(desc[r][c], "yellow", highlight=True)
                elif desc[r][c] in "FTAL":
                    desc[r][c] = utils.colorize(desc[r][c], "green", highlight=True)
                else:
                    desc[row][col] = utils.colorize(desc[row][col], "magenta", highlight=True)
        desc[0][0] = utils.colorize(desc[0][0], "yellow", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Down', 'Right', 'Up', 'Left'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")
        """for ball in self.collected_dragon_balls:
            ball_row, ball_col = self.to_row_col(ball)
            desc[ball_row][ball_col] = utils.colorize(desc[ball_row][ball_col], "green", highlight=True)"""

        with closing(outfile):
            return outfile.getvalue()
