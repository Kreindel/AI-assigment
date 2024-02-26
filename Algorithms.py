from curses import halfdelay
from math import sqrt
import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

NO_ACTION = -1

class Node():
    def __init__(self, state, parent, action=NO_ACTION, cost =0, h_weight=0, env=None, is_hole=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = cost
        if parent:
            self.g += parent.g
        self.h_weight = h_weight
        self.env = env
        self.is_hole = is_hole

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def update(self, node):
        self.state = node.state
        self.parent = node.parent
        self.action = node.action
        self.g = node.g

    def manhatan(self, s):
        s1 = self.env.to_row_col(self.state)
        s2 = self.env.to_row_col(s)
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def h_MSAP(self):
        dist = []
        d1_collected = self.state[1]
        d2_collected = self.state[2]

        if not d1_collected:
            dist.append(self.manhatan(self.env.d1))
        if not d2_collected:
            dist.append(self.manhatan(self.env.d2))
        for goal in self.env.goals:
            dist.append(self.manhatan(goal))

        return min(dist)

    def euclidean(self, s):
        s1 = self.env.to_row_col(self.state)
        s2 = self.env.to_row_col(s)
        return sqrt((s1[0]-s2[0])**2 + (s1[1] - s2[1])**2)

    def h_dist(self):
        dist = []
        d1_collected = self.state[1]
        d2_collected = self.state[2]

        if not d1_collected:
            dist.append(self.euclidean(self.env.d1))
        if not d2_collected:
            dist.append(self.euclidean(self.env.d2))
        for goal in self.env.goals:
            dist.append(self.euclidean(goal))

        return min(dist)

    def f(self):
        return self.h_weight*self.h_MSAP() + (1-self.h_weight)*self.g

    def __str__(self):
        return f"({self.state[0]}, {self.state[1]}, {self.state[2]})"
        

class Agent():
    def __init__(self):
        self.env = None
        
    def solution(self, node, expanded):
        actions = []
        total_cost = node.g
        
        while node.action != NO_ACTION:
            actions.insert(0, node.action)
            node = node.parent
            
        return (actions, total_cost, expanded)

    def succ(self, node):
        self.env.collected_dragon_ball = (node.state[1], node.state[2])    # Update the board.
        info = self.env.succ(node.state)
        for action in info:
            next_state = info[action][0]
            idx, d1, d2 = next_state[0], node.state[1], node.state[2]
            if self.env.d1[0] == next_state[0]:
                d1 = True
            if self.env.d2[0] == next_state[0]:
                d2 = True
            info[action] = ((idx, d1, d2), info[action][1], info[action][2])
        
        return info

class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        expanded = 0
        
        node = Node(self.env.get_initial_state(), None)
        if self.env.is_final_state(node.state):
            return self.solution(node, expanded)
        
        open = [node]
        close = []
        while open:
            node = open.pop()
            close.append(node.state)
            expanded += 1
            if node.is_hole:
                continue
            for action, (next_state, cost, terminated) in self.succ(node).items():
                child = Node(next_state, node, action, cost)
                if child.state not in close and child not in open:
                    if self.env.is_final_state(child.state):
                        return self.solution(child, expanded)
                    elif terminated:
                        child.is_hole = True
                    open.insert(0, child)
              

        print("Couldn't find solution")


class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.h_weight = 0

    def get_f(self, h, g):
        return self.h_weight*h + (1-self.h_weight)*g

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.h_weight = h_weight
        expanded = 0

        node = Node(self.env.get_initial_state(), None, h_weight=h_weight, env=env)
        open = heapdict.heapdict()
        open[node] = (node.f(), node.state[0])
        close = dict()
        while open:
            node = open.popitem()[0]
            close[node.state] = node
            if self.env.is_final_state(node.state):
                return self.solution(node, expanded)
            expanded += 1
            for action, (next_state, cost, terminated) in self.succ(node).items():
                child = Node(next_state, node, action, cost, h_weight, env)
                if child not in open and child.state not in close:
                    open[child] = (child.f(), child.state[0])
                elif child in open:
                    if child.f() < open[child][0]:
                        open.pop(child)
                        open[child] = (child.f(), child.state[0])
                else:
                    node_curr = close[child.state]
                    if child.f() < node_curr.f():
                        open[child] = (child.f(), child.state[0])
                        del close[node_curr.state]



class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 0

    def focal(self, open):
        thresh_hold = (1+self.epsilon) * open.peekitem()[0].f()
        focal = heapdict.heapdict()
        for n in open.keys():
            if n.f() <= thresh_hold:
                focal[n] = (n.g, n.state[0])

        return focal


    def pop_focal(self, open):
        focal = self.focal(open)
        node = focal.popitem()[0]
        del open[node]
        return node


    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.epsilon = epsilon
        expanded = 0

        node = Node(self.env.get_initial_state(), None, h_weight=0.5, env=env)
        open = heapdict.heapdict()
        open[node] = node.f()
        close = dict()
        while open:
            node = self.pop_focal(open)
            close[node.state] = node
            if self.env.is_final_state(node.state):
                return self.solution(node, expanded)
            expanded += 1
            for action, (next_state, cost, terminated) in self.succ(node).items():
                child = Node(next_state, node, action, cost, h_weight=0.5, env=env)
                if child not in open and child.state not in close:
                    open[child] = child.f()
                elif child in open:
                    if child.f() < open[child]:
                        open.pop(child)
                        open[child] = child.f()
                else:
                    node_curr = close[child.state]
                    if child.f() < node_curr.f():
                        open[child] = child.f()
                        del close[node_curr.state]

class AStarEpsilonAgent2(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 0

    def focal(self, open):
        thresh_hold = (1+self.epsilon) * open.peekitem()[0].f()
        focal = heapdict.heapdict()
        for n in open.keys():
            if n.f() <= thresh_hold:
                focal[n] = (n.h_dist(), n.state[0])

        return focal

    def pop_focal(self, open):
        focal = self.focal(open)
        node = focal.popitem()[0]
        del open[node]
        return node


    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.epsilon = epsilon
        expanded = 0

        node = Node(self.env.get_initial_state(), None, h_weight=0.5, env=env)
        open = heapdict.heapdict()
        open[node] = node.f()
        close = dict()
        while open:
            node = self.pop_focal(open)
            close[node.state] = node
            if self.env.is_final_state(node.state):
                return self.solution(node, expanded)
            expanded += 1
            for action, (next_state, cost, terminated) in self.succ(node).items():
                child = Node(next_state, node, action, cost, h_weight=0.5, env=env)
                if child not in open and child.state not in close:
                    open[child] = child.f()
                elif child in open:
                    if child.f() < open[child]:
                        open.pop(child)
                        open[child] = child.f()
                else:
                    node_curr = close[child.state]
                    if child.f() < node_curr.f():
                        open[child] = child.f()
                        del close[node_curr.state]
