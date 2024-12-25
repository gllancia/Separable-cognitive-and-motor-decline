import numpy as np
import networkx as nx
from scipy.special import softmax
from baseAgent import BaseAgent


class ShortestPathAgent(BaseAgent):
    def __init__(self, graphManager, goal_state, beta=1):
        super().__init__(graphManager)
        
        self.goal_state = goal_state
        self.beta = beta
        
        self.create_distance_action_matrix()
            
            
    def create_distance_action_matrix(self):
        self.distances_from_goal = nx.shortest_path_length(self.graphManager.O, target=(*self.goal_state, np.random.randint(8)), weight='weight')
        
        self.distance_action_matrix = -np.ones((len(self.graphManager.O.nodes), 8))
        
        for i,(x,y,o) in enumerate(self.graphManager.O.nodes):
            for action in [(o-1)%8, o, (o+1)%8]:
                new_state, is_valid_state = super().get_new_state_from_action(self.graphManager.node2index[(x,y)], action)
                if is_valid_state and new_state in self.graphManager.G.neighbors((x,y)):
                        self.distance_action_matrix[i,action] = self.distances_from_goal[(*new_state,action)]
                else:
                    self.distance_action_matrix[i,action] = self.distances_from_goal[(x,y,o)]
                    
        self.alternative_distances_from_goal = nx.shortest_path_length(self.graphManager.G, target=self.goal_state, weight='weight')
        self.alternative_distance_action_matrix = -np.ones((len(self.graphManager.G.nodes), 8))
        
        for i,state in enumerate(self.graphManager.G.nodes):
            for action in range(8):
                new_state, is_valid_state = super().get_new_state_from_action(i, action)
                if is_valid_state and new_state in self.graphManager.G.neighbors(state):
                        self.alternative_distance_action_matrix[i,action] = self.alternative_distances_from_goal[new_state]
                else:
                    self.alternative_distance_action_matrix[i,action] = self.alternative_distances_from_goal[state]


         
    def get_action(self, state, orientation, beta=None):
        left, center, right = (orientation-1)%8, orientation, (orientation+1)%8
        Ostate = self.graphManager.Onode2index[(*self.graphManager.index2node[state],orientation)]
        actions = -self.distance_action_matrix[Ostate,[left,center,right]]
        
        if actions.sum() == 0:
            actions = np.ones(3)
        #actions = actions/np.sum(actions)
        if beta is None:
            beta = self.beta
        actions = softmax(beta*actions)
        
        chosen_action = np.random.choice([left,center,right],p=actions)
        return chosen_action, [left,center,right], actions
    
    
    def alternative_get_action(self, state, orientation, beta=None):
        distances = -self.alternative_distance_action_matrix[state,:]
        
        if distances.sum() == 0:
            distances = np.ones(8)
        #distances = distances/np.sum(distances)
        if beta is None:
            beta = self.beta
            
        #sdistances = -np.log(distances+1)
        actions = softmax(beta*distances)
        
        chosen_action = np.random.choice(range(8),p=actions)
        return chosen_action, range(8), actions 



    def evaluate_likelihood_of_action(self, state, orientation, action, use_alternative=False, beta=None):
        if not use_alternative:
            chosen_action, possibile_actions, ps = self.get_action(state, orientation, beta=beta)
        else:
            chosen_action, possibile_actions, ps = self.alternative_get_action(state, orientation, beta=beta)
        if action in possibile_actions:
            return ps[possibile_actions.index(action)]
        else:
            return np.nan
        
        
        
    def evaluate_likelihood_of_trajectory(self, states, orientations, use_alternative=False, beta=None):
        likelihood = []
        for i in range(len(states)-1):
            action = super().get_action_from_movement(states[i], states[i+1])
            likelihood.append(self.evaluate_likelihood_of_action(states[i],orientations[i],action,use_alternative,beta=beta))
        return np.array(likelihood)