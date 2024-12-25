import numpy as np
import json
import importlib
from scipy.special import softmax

import graph_manager
_ = importlib.reload(graph_manager)


class Agent():
    def __init__(self, graphManager) -> None:
        self.graphManager = graphManager
        self.actions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]])
        
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
        
        return self.graphManager.node2index[state], (orientation + 1)%8
    


class VisibilityAgent(Agent):
    def __init__(self, graphManager, level):
        super().__init__(graphManager)
        
        with open('./data/data_processed/visibilities.json', 'r') as fp:
            self.visibility = np.array(json.load(fp)[str(level)])
            
        self.visibleNodes = np.zeros((len(graphManager.G.nodes), 8, len(graphManager.G.nodes)))
        self.visibilityByOrientation = np.empty((len(graphManager.G.nodes), 8))
        

        self.visitation_matrix = np.zeros(len(graphManager.G.nodes))
        
        self.create_vis_cone_array(graphManager)
        
        self.update_visibility(graphManager)
        
    def create_vis_cone_array(self, graphManager):
        self.visConeArray = np.zeros((len(graphManager.G.nodes), 8, len(graphManager.G.nodes)))
        for s,_ in enumerate(graphManager.G.nodes):
            for o in [0,1,2,3,4,-3,-2,-1]:
                self.visConeArray[s,o%8] = graphManager.get_visibility_cone_array(s, np.deg2rad(o*45), np.deg2rad(45))

    def update_visibility(self, graphManager):
        for s,_ in enumerate(graphManager.G.nodes):
            for o in [0,1,2,3,4,-3,-2,-1]:
                #visConeArray = graphManager.get_visibility_cone_array(s, np.deg2rad(o*45), np.deg2rad(45))
                
                self.visibleNodes[s,o%8] = self.visibility[s]*self.visConeArray[s,o%8]*np.exp(-self.visitation_matrix)
                self.visibilityByOrientation[s,o%8] = np.sum(self.visibility[s]*self.visConeArray[s,o%8]*np.exp(-self.visitation_matrix))
        

    def get_action_from_visibility(self, state, orientation):
        left, center, right = (orientation-1)%8, orientation, (orientation+1)%8
        actions = self.visibilityByOrientation[state,[left,center,right]]
        
        if actions.sum() == 0:
            actions = np.ones(3)
        actions = actions/np.sum(actions)
        actions = softmax(5*actions)
        return np.random.choice([left,center,right],p=actions), [left,center,right], actions
    
    def updateOccupancy(self, state, graphManager, updateVisibility=False):
        states_checked = []
        penalty = 1.
        self.visitation_matrix[state] += penalty
        states_checked.append(state)
        
        for neighbor in graphManager.G.neighbors(graphManager.index2node[state]):
            neighbor_idx = graphManager.node2index[neighbor]
            if neighbor_idx in states_checked:
                continue
            self.visitation_matrix[neighbor_idx] += penalty/2.
            states_checked.append(neighbor_idx)
            
            for neighbor2 in graphManager.G.neighbors(neighbor):
                neighbor2_idx = graphManager.node2index[neighbor2]
                if neighbor2_idx in states_checked:
                    continue
                self.visitation_matrix[neighbor2_idx] += penalty/4.
                states_checked.append(neighbor2_idx)
                
                for neighbor3 in graphManager.G.neighbors(neighbor2):
                    neighbor3_idx = graphManager.node2index[neighbor3]
                    if neighbor3_idx in states_checked:
                        continue
                    self.visitation_matrix[neighbor3_idx] += penalty/4.
                    states_checked.append(neighbor3_idx)
                    
                    for neighbor4 in graphManager.G.neighbors(neighbor3):
                        neighbor4_idx = graphManager.node2index[neighbor4]
                        if neighbor4_idx in states_checked:
                            continue
                        self.visitation_matrix[neighbor4_idx] += penalty/4.
                        states_checked.append(neighbor4_idx)
                        
                        
                
        if updateVisibility:
            
            #self.visitation_matrix = self.visitation_matrix/np.max(self.visitation_matrix)
            self.update_visibility(graphManager)
        
    