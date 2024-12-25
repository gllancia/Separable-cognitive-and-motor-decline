import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from copy import deepcopy
from scipy.spatial.distance import euclidean


def extract_trajectory(DF_row, skip_pauses=True):
    full_path = json.loads(DF_row)
    rs = np.array(full_path["r"])%8
    idxs = list(range(len(rs)))
    
    if skip_pauses:
        diff = np.sqrt(np.diff(full_path["x"])**2 + np.diff(full_path["y"])**2)
        idxs = [i for i in range(len(diff)) if diff[i] > 0.5]
        
        
    
    return np.array(full_path["x"])[idxs], np.array(full_path["y"])[idxs], np.array(rs.tolist())[idxs]


def plot_trajectory(trajectory, graphManager, sizes=None, 
                    color="b", plot_imshow=True):
    xs, ys, os = np.array(trajectory).T
    
    if plot_imshow:
        plt.imshow(graphManager.map_matrix, cmap=cm.gray, zorder=0,
               origin='lower')
    
    if sizes is not None:
        for x,y,s in zip(xs,ys,sizes):
            if not np.isnan(s):
                plt.scatter(x,y, s=s, zorder=5, alpha=0.5, c=color)
            else:
                plt.scatter(x,y, s=30, zorder=5, alpha=0.5, c="black")
    else:
        plt.scatter(xs, ys, s=10, zorder=5, alpha=0.5)
        

def extract_goal_touch_index(xs,ys,goals,d_threshold=3):
    goals = deepcopy(goals)
    goal_touches = []
    id_goal = 0
    for i,(x,y) in enumerate(zip(xs,ys)):
        if euclidean((x,y),goals[id_goal]) < d_threshold:
            if i+d_threshold < len(xs):
                goal_touches.append(i+d_threshold)
            else:
                goal_touches.append(len(xs)-1)
            id_goal += 1
            if id_goal == len(goals):
                break
    return goal_touches


def get_landmarks(map_filtered):
    landmark_xs, landmark_ys = [], []
    
    landmark_xs.append(map_filtered["start"][0])
    landmark_ys.append(map_filtered["start"][1])
    
    for x,y in map_filtered["goals"]:
        landmark_xs.append(x)
        landmark_ys.append(y)
    return landmark_xs, landmark_ys

def graph_to_imshow(map_filtered):
    H,W = map_filtered["size"]
    map_matrix = np.zeros((H,W))
    
    for x,y in map_filtered["nodes"]:
        map_matrix[y,x] = 1
    return map_matrix


def nodes2indexes(xs, ys, graphManager):
    return np.array([graphManager.node2index[(x,y)] for x,y in zip(xs,ys)])

def indexes2nodes(indexes, graphManager):
    return [tuple(graphManager.index2node[i]) for i in indexes]