import os
import sys
sys.path.insert(0, os.path.abspath('./agents'))
sys.path.insert(0, os.path.abspath('./model'))

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
import os
import networkx as nx
import importlib
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from copy import deepcopy
import json
from tqdm.auto import tqdm
from scipy.optimize import minimize


import graph_manager
_ = importlib.reload(graph_manager)
from graph_manager import GraphManager
import visibilityAgent
_ = importlib.reload(visibilityAgent)
from visibilityAgent import VisibilityAgent
import shortestPathAgent
_ = importlib.reload(shortestPathAgent)
from shortestPathAgent import ShortestPathAgent

    
def nodes2indexes(xs, ys, graphManager):
    return np.array([graphManager.node2index[(x,y)] for x,y in zip(xs,ys)])

def indexes2nodes(indexes, graphManager):
    return [tuple(graphManager.index2node[i]) for i in indexes]


def old_extract_trajectory(DF_row):
    full_path = json.loads(DF_row)
    rs = np.array(full_path["r"])%8
    return full_path["x"], full_path["y"], rs.tolist()

def extract_trajectory(DF_row, skip_pauses=True):
    full_path = json.loads(DF_row)
    rs = np.array(full_path["r"])%8
    idxs = list(range(len(rs)))
    
    if skip_pauses:
        diff = np.sqrt(np.diff(full_path["x"])**2 + np.diff(full_path["y"])**2)
        idxs = [i for i in range(len(diff)) if diff[i] > 0.5]
    
    return np.array(full_path["x"])[idxs], np.array(full_path["y"])[idxs], np.array(rs.tolist())[idxs]

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

def init_agents(graphManager, visibility, level, betaVis, betaSP):
    visAgent = VisibilityAgent(graphManager, level, visibility=np.array(visibility[str(level)]), 
                           path_to_visibility="", beta=betaVis)
    
    spAgent0 = ShortestPathAgent(graphManager, graphManager.index2node[graphManager.goals[0]], beta=betaSP)
    spAgent1 = ShortestPathAgent(graphManager, graphManager.index2node[graphManager.goals[1]], beta=betaSP)
    spAgent2 = ShortestPathAgent(graphManager, graphManager.index2node[graphManager.goals[2]], beta=betaSP)
    
    agents = [visAgent, spAgent0, spAgent1, spAgent2]
    
    if len(graphManager.goals) > 3:
        spAgent3 = ShortestPathAgent(graphManager, graphManager.index2node[graphManager.goals[3]], beta=betaSP)
        agents.append(spAgent3)
        
    return agents
    
def fun(beta, agent, start, stop, xs, ys, rs, use_alternative, use_negative=False, return_full_trajectory=False):
    likelihood = agent.evaluate_likelihood_of_trajectory(nodes2indexes(xs[start:stop-1], ys[start:stop-1], graphManager=graphManager), 
                        rs[start:stop-1], use_alternative=use_alternative, beta=beta)
    
    if return_full_trajectory:
        return -np.log(likelihood)
    
    if not use_negative:
        return -np.sum(-np.log(likelihood), where=~np.isnan(likelihood))
    else:
        return np.sum(-np.log(likelihood), where=~np.isnan(likelihood))


def eval_multiple_ll(user_number, DF, level, graphManager, agents):
    use_alternative = False

    xs, ys, rs = extract_trajectory(DF[DF.level_id==level].iloc[user_number]["path"])
    idxs = extract_goal_touch_index(xs, ys, indexes2nodes(graphManager.goals, graphManager=graphManager))
    idxs.insert(0,0)
    
    Ls = []
    for agent in agents:
        L = []
        for j in range(len(idxs)-1):
            start = idxs[j]
            stop = idxs[j+1]
                
            res = minimize(fun, x0=10, bounds=[(0.01, 15)], 
                args=(agent, start, stop, xs, ys, rs, use_alternative, True))
                
            beta = res.x    
            L.append(fun(beta, agent, start, stop, xs, ys, rs, use_alternative, return_full_trajectory=True))
            #L.append(-res.fun)
        Ls.append(L)

    
    return Ls, idxs
    
    

    
    
if __name__ == "__main__":
    which_level_to_run = int(sys.argv[1])
    
    print("Loading dataset")
    DF = pd.read_csv('../DataProcessed/Full_Dataset_Interpolated.csv')
    
    n_subjects = len(DF.user_id.unique())

    print("Loading maps")
    with open('../DataProcessed/maps_filtered.json', 'r') as fp:
        maps_filtered = json.load(fp)
    
    print("Loading visibility")
    with open("../DataProcessed/visibilities.json", 'r') as fp:
        visibility = json.load(fp)
        
    levels = [ 1,  2,  6,  7, 11, 12, 13, 16, 17, 21, 26, 27, 31, 32, 42, 43, 46]
    
    levels = [levels[which_level_to_run]]
    
    n_agents = 5
    n_goals = 4
    
    likelihoods = -np.ones((n_subjects, len(levels), n_agents, n_goals), dtype="object")
    
    
    print("Starting evaluation of level: ", levels[0])
    for level in levels:
        level_index = levels.index(level)
        
        if level in [1,2,6]:
            continue
        
        graphManager = GraphManager(maps_filtered[str(level)]["nodes"], 
                                maps_filtered[str(level)]["edges"], 
                                maps_filtered[str(level)]["goals"],
                                size=maps_filtered[str(level)]["size"])
        graphManager.extend_graph_with_orientation()
        
        agents = init_agents(graphManager, visibility, level, betaVis=10, betaSP=10)
        
        for i_subject, subject in enumerate(range(n_subjects)):
            if i_subject % 1000 == 0:
                print(f"Level {level}: {np.round(i_subject/n_subjects * 100, 0)}%")
            Ls, idxs = eval_multiple_ll(subject, DF, level, graphManager, agents)
            for LId, L in enumerate(Ls):
                for g in range(len(idxs)-1):
                    likelihoods[subject, level_index, LId, g] = np.array(L[g])
                    
                    
        np.save(f"likelihoods_level_{which_level_to_run}.npy", likelihoods)
    
