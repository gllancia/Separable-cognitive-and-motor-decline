import networkx as nx
from scipy.spatial.distance import cdist
import numpy as np



class GraphManager():
    """
    7   0   1       NW   N   NE
    6   8   2       W    I    E
    5   4   3       SW   S   SE
    """
    
    def __init__(self, nodes, edges, goals, allow_diagonal_actions=True, size=(0,0)):
        self.load_graph(nodes, edges, allow_diagonal_actions, goals)
    
        self.get_first_neightbour_distance_matrix()
        self.get_distance_matrix()
        self.get_angle_matrix()
        if size != (0,0):
            self.get_map_matrix(size)
    
    def load_graph(self, nodes, edges, allow_diagonal_actions,goals):
        self.G = nx.Graph()
        
        for node in nodes:
            self.G.add_node(tuple(node), pos=(node[0], node[1]))
        
        for edge in edges:
            self.G.add_edge(tuple(edge[0]), tuple(edge[1]), 
                            weight=np.sqrt((edge[0][0]-edge[1][0])**2 + (edge[0][1]-edge[1][1])**2))
        
        if allow_diagonal_actions:
            self.actions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,0]])
        else:
            self.actions = np.array([[0,1], [1,0], [0,-1], [-1,0], [0,0]])
        self.n_actions = len(self.actions)
        
        self.node2index = {node:index for index,node in enumerate(self.G.nodes)}
        self.index2node = {index:node for index,node in enumerate(self.G.nodes)}
        self.goals = [self.node2index[tuple(goal)] for goal in goals]
            
            
    def get_first_neightbour_distance_matrix(self):
        self.distance_matrix = -np.ones((len(self.G.nodes), len(self.G.nodes), self.n_actions)) 
                #(n_states, n_states, n_actions), 0 if no transition, distance if transition
        
        for a,nodeA in enumerate(self.G.nodes):
            for b,nodeB in enumerate(self.G.nodes):
                edge = (nodeA, nodeB)
                if edge in self.G.edges:
                    displacement = list(np.array(nodeB) - np.array(nodeA)) 
                    action = self.actions.tolist().index(displacement)
                    self.distance_matrix[b, a, action] = np.sqrt((nodeA[0]-nodeB[0])**2 + (nodeA[1]-nodeB[1])**2)
            
        #when in goal state, every action takes you back to the same state
        #for goal in self.goals:
        #    for state in range(len(self.G.nodes)):
        #        self.p_s_s_a[state, goal] = 0.
        #    self.p_s_s_a[goal, goal] = 1.
        
    
    def get_distance_matrix(self):
        self.distance_matrix = cdist(self.G.nodes,self.G.nodes)    
     
    def get_angle_between_states(self, nodeA, nodeB):
        """
        Returns the angle between two states in radians, in the range [-pi, pi].
        Zero is the positive y-axis, and the angle increases clockwise.
        """
        dx = nodeB[0] - nodeA[0]
        dy = nodeB[1] - nodeA[1]
        return np.arctan2(dx, dy)
    
    def get_angle_matrix(self):
        self.angle_matrix = np.zeros((len(self.G.nodes), len(self.G.nodes)))

        for i,node in enumerate(self.G.nodes):
            for j, node2 in enumerate(self.G.nodes):
                self.angle_matrix[i, j] = self.get_angle_between_states(node, node2)
                
    def is_in_cone_of_view(self, direction_angle, target_angle, view_cone_angle, is_in_radians=True):
        if not is_in_radians:
            direction_angle = np.deg2rad(direction_angle)
            view_cone_angle = np.deg2rad(view_cone_angle)
            target_angle = np.deg2rad(target_angle)
            
        if  - view_cone_angle <= 2*(target_angle - direction_angle) <= view_cone_angle:
            return True
        
        elif direction_angle + view_cone_angle/2. > np.pi:
            if -np.pi <= target_angle <= direction_angle + view_cone_angle/2. - 2*np.pi:
                return True
               
        elif direction_angle - view_cone_angle/2. < -np.pi:
            if direction_angle - view_cone_angle/2. + 2*np.pi <= target_angle <= np.pi:
                return True
            
        else:
            return False
        
    def get_visibility_cone_array(self, stateIndex, direction_angle, view_cone_angle):
        visibility_cone_array = np.zeros(len(self.G.nodes))
        for targetStateIndex in range(len(self.G.nodes)):
            if self.is_in_cone_of_view(direction_angle, self.angle_matrix[stateIndex, targetStateIndex], view_cone_angle):
                visibility_cone_array[targetStateIndex] = 1
            else:
                visibility_cone_array[targetStateIndex] = 0
        
        return visibility_cone_array
    
    def get_map_matrix(self, map_size):
        self.H,self.W = map_size
        map_matrix = np.zeros((self.H,self.W))
        
        for x,y in self.G.nodes:
            map_matrix[y,x] = 1
            
        self.map_matrix = map_matrix
        return map_matrix
    
    
    def extend_graph_with_orientation(self):
        diff2action = {(0,1):0, (1,1):1, (1,0):2, (1,-1):3, (0,-1):4, (-1,-1):5, (-1,0):6, (-1,1):7, (0,0):-1}
        
        self.O = nx.DiGraph()
        for node in self.G.nodes:
            for orientation in range(8):
                self.O.add_node((*node, orientation))
                
        for nodeA in self.G.nodes:
            for nodeB in self.G.nodes:
                if nodeA != nodeB and nodeB in self.G.neighbors(nodeA):
                    diff = np.array(nodeB) - np.array(nodeA)
                    orientation = diff2action[tuple(diff)]
                    
                    self.O.add_edge((*nodeA, (orientation-1)%8), (*nodeB, orientation), weight=self.G[nodeA][nodeB]["weight"])
                    self.O.add_edge((*nodeA, orientation), (*nodeB, orientation), weight=self.G[nodeA][nodeB]["weight"])
                    self.O.add_edge((*nodeA, (orientation+1)%8), (*nodeB, orientation), weight=self.G[nodeA][nodeB]["weight"])
        
        for node in self.O.nodes:
            if len(list(self.O.neighbors(node))) == 0:
                for action in range(8):
                    self.O.add_edge(node, (node[0], node[1], action), weight=0)
                    self.O.add_edge((node[0], node[1], action), node, weight=0)
                    
        self.Onode2index = {node:index for index,node in enumerate(self.O.nodes)}
        self.Oindex2node = {index:node for index,node in enumerate(self.O.nodes)}
                    