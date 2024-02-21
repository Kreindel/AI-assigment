import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

NO_ACTION = -1

class Node():
    def __init__(self, state, parent, action = NO_ACTION, cost = 0):
        #print(f"Create Node: state=({state[0]}, {state[1]}, {state[2]}), action={action}, cost = {cost}.")
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __eq__(self, other):
        return self.state == other.state
        
class Agent():
    def __init__(self):
        self.env = None
        
    def solution(self, node, expanded):
        actions = []
        total_cost = 0
        
        while node.action != NO_ACTION:
            actions.insert(0, node.action)
            total_cost += node.cost
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
            print(f"state=({node.state[0]}, {node.state[1]}, {node.state[2]})")
            print(f"{node.state[0]}->", end="")
            for action, (next_state, cost, terminated) in self.succ(node).items():
                child = Node(next_state, node, action, cost)
                if child.state not in close and child not in open:
                    if self.env.is_final_state(child.state):
                        return self.solution(child, expanded)
                    elif not terminated:
                        open.insert(0, child)

        print("Couldn't find solution")

class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
