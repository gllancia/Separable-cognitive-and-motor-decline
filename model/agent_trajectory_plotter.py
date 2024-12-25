import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
from scipy.spatial.distance import pdist
from graph_manager import GraphManager
import pandas as pd
import time

from agents import VisibilityAgent



def graph_to_imshow(map_filtered):
    H,W = map_filtered["size"]
    map_matrix = np.zeros((H,W))
    
    for x,y in map_filtered["nodes"]:
        map_matrix[y,x] = 1
    return map_matrix

def get_landmarks(map_filtered):
    landmark_xs, landmark_ys = [], []
    
    landmark_xs.append(map_filtered["start"][0])
    landmark_ys.append(map_filtered["start"][1])
    
    for x,y in map_filtered["goals"]:
        landmark_xs.append(x)
        landmark_ys.append(y)
    return landmark_xs, landmark_ys

def draw_cone(x,y,angle,angle_width,length=1):
    plt.plot([x,x+length*np.sin(angle-angle_width/2)],[y,y+length*np.cos(angle-angle_width/2)], color='r')
    plt.plot([x,x+length*np.sin(angle+angle_width/2)],[y,y+length*np.cos(angle+angle_width/2)], color='r')
    
def get_visibility_matrix(map_filtered, visible_nodes, state, orientation):
    H,W = map_filtered["size"]
    visibility_matrix = np.zeros((H,W))
    
    vNodes = visible_nodes[state,(orientation-1)%8]+visible_nodes[state,(orientation)%8]+visible_nodes[state,(orientation+1)%8]
    #vNodes[vNodes > 1] = 1
    vNodes[vNodes > 1] /= 3
    if vNodes.sum() > 0:
        vNodesIdx = np.where(vNodes > 0)[0]
    else:
        vNodesIdx = []
    
    for i,(x,y) in enumerate(map_filtered["nodes"]):
        if i in vNodesIdx:
            visibility_matrix[y,x] = vNodes[i]
            
        else:
            visibility_matrix[y,x] = 1
    return visibility_matrix



if __name__ == '__main__':
    level = "42"
    
    with open('./data/data_processed/maps_filtered.json', 'r') as fp:
        maps_filtered = json.load(fp)
        
    graphManager = GraphManager(maps_filtered[level]["nodes"], 
                                maps_filtered[level]["edges"], 
                                maps_filtered[level]["goals"])
    map_matrix = graph_to_imshow(maps_filtered[level])
    landmark_xs, landmark_ys = get_landmarks(maps_filtered[level])
    
    agent = VisibilityAgent(graphManager, level)
    state = graphManager.node2index[tuple(maps_filtered[level]["start"])]
    orientation = 0
    
      
    plt.ion()
    for _ in range(1000):
        plt.scatter(landmark_xs, landmark_ys, s=100, c='red',zorder=-5)
        plt.text(landmark_xs[0], landmark_ys[0], "S", fontsize=20, zorder=-5)
        for i in range(1, len(landmark_xs)):
            plt.text(landmark_xs[i], landmark_ys[i], i, fontsize=20, zorder=-5)
        #plt.imshow(map_matrix, cmap='gray', zorder=-15, origin='lower')
        
        action, actions, ps = agent.get_action_from_visibility(state, orientation)
        state, orientation = agent.move(state, orientation, action)
        agent.updateOccupancy(state, graphManager, (_%20 == 0))
        draw_cone(*graphManager.index2node[state], np.deg2rad(orientation*45), np.deg2rad(135), length=5)
        visibility_matrix = get_visibility_matrix(maps_filtered[level], agent.visibleNodes, state, orientation)
        plt.imshow(visibility_matrix, cmap='gray', zorder=-10, origin='lower')
        
        x,y = graphManager.index2node[state]
        plt.scatter(x, y, s=30, zorder=5)
        
        #plt.arrow(x,y, np.sin(orientation*np.pi/4), np.cos(orientation*np.pi/4), width=0.1, zorder=5)
        for a,p in zip(actions,ps):
            plt.arrow(x, y, 3*p*np.sin(a*np.pi/4), 3*p*np.cos(a*np.pi/4), width=0.1, zorder=5)
            
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-0.5,maps_filtered[level]["size"][1]-0.5)
        plt.ylim(-0.5,maps_filtered[level]["size"][0]-0.5)
        plt.draw()
        #plt.savefig("./model/out/move_figs/000{}.png".format(_))
        plt.savefig("./model/out/move_figs/{:04d}.png".format(_))
        plt.pause(0.005)
        plt.clf()
        
        print(_, end="\r")