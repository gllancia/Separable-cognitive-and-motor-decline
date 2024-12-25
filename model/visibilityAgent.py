import numpy as np
import json
from scipy.special import softmax
from .baseAgent import BaseAgent


class VisibilityAgent(BaseAgent):
    def __init__(self, 
                 graphManager, 
                 level, 
                 path_to_visibility='./data/data_processed/visibilities.json',
                 visibility=None, 
                 beta=5):
        super().__init__(graphManager)
        
        if path_to_visibility == "":
            self.visibility = visibility
        else:
            with open(path_to_visibility, 'r') as fp:
                self.visibility = np.array(json.load(fp)[str(level)])
            
        self.visibleNodes = np.zeros((len(graphManager.G.nodes), 8, len(graphManager.G.nodes)))
        self.visibilityByOrientation = np.empty((len(graphManager.G.nodes), 8))
        self.beta = beta

        
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
                
                self.visibleNodes[s,o%8] = self.visibility[s]*self.visConeArray[s,o%8]
                self.visibilityByOrientation[s,o%8] = np.sum(self.visibility[s]*self.visConeArray[s,o%8])
        

    def get_action(self, state, orientation, beta=None):
        left, center, right = (orientation-1)%8, orientation, (orientation+1)%8
        vis = self.visibilityByOrientation[state,[left,center,right]]
        
        if vis.sum() == 0:
            vis = np.ones(3)
        vis = vis/np.sum(vis)
        if beta is None:
            beta = self.beta
        actions = softmax(beta*vis)
        
        chosen_action = np.random.choice([left,center,right],p=actions)
        return chosen_action, [left,center,right], actions
    
    
    def alternative_get_action(self, state, orientation, beta=None):
        vis = self.visibilityByOrientation[state,:]
        
        if vis.sum() == 0:
            vis = np.ones(8)
        vis = vis/np.sum(vis)
        if beta is None:
            beta = self.beta
        actions = softmax(beta*vis)
        
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
            likelihood.append(self.evaluate_likelihood_of_action(states[i],orientations[i],action, use_alternative=use_alternative, beta=beta))
        return np.array(likelihood)