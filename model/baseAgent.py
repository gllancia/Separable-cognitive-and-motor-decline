import numpy as np



class BaseAgent():
    def __init__(self, graphManager):
        self.graphManager = graphManager
        self.actions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]])
        self.diff2action = {(0,1):0, (1,1):1, (1,0):2, (1,-1):3, (0,-1):4, (-1,-1):5, (-1,0):6, (-1,1):7, (0,0):-1}
        self.action2diff = {0:(0,1), 1:(1,1), 2:(1,0), 3:(1,-1), 4:(0,-1), 5:(-1,-1), 6:(-1,0), 7:(-1,1), -1:(0,0)}
        
    def move(self, state, orientation, action):
        da = np.abs(orientation - action)
        if da > 4:
            da = 8 - da
        if da > 1:
            return state, (orientation + 1)%8
        
        state = self.graphManager.index2node[state]
        next_state = tuple(np.array(state) + self.actions[action])
        if next_state in self.graphManager.G.nodes:
            if next_state in self.graphManager.G.neighbors(state):
                return self.graphManager.node2index[next_state], action
        
        random_dir = np.random.choice([1,-1])
        return self.graphManager.node2index[state], (orientation + random_dir)%8
    
    
    def get_action_from_movement(self, state, next_state):
        state = self.graphManager.index2node[state]
        next_state = self.graphManager.index2node[next_state]
        diff = tuple(np.array(next_state) - np.array(state))
        return self.diff2action[diff]
    
    
    def get_new_state_from_action(self, state, action):
        state = self.graphManager.index2node[state]
        new_state = tuple(np.array(state) + self.actions[action])
        return new_state, new_state in self.graphManager.G.nodes