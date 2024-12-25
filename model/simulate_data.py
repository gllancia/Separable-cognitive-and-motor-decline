import numpy as np
import pandas as pd
import json
from utils import *
from copy import deepcopy
from graph_manager import GraphManager
from shortestPathAgent import ShortestPathAgent
import sys
import os


if __name__ == "__main__":
    level = int(sys.argv[1])

    n_synth_users_per_beta = 500
    max_length = 1000
    beta_values = [0.01, 0.1, 0.5, 1, 5]

    levels = [ 1,  2,  6,  7, 11, 12, 13, 16, 17, 21, 26, 27, 31, 32, 42, 43, 46]

    with open('../data/data_processed/maps_filtered.json', 'r') as fp:
        maps_filtered = json.load(fp)

    synth_DF = []

    graphManager = GraphManager(maps_filtered[str(level)]["nodes"], 
                                maps_filtered[str(level)]["edges"], 
                                maps_filtered[str(level)]["goals"],
                                size=maps_filtered[str(level)]["size"])
    graphManager.extend_graph_with_orientation()

    spAgents = [ShortestPathAgent(graphManager, graphManager.index2node[graphManager.goals[g]], beta=5) 
            for g in range(len(graphManager.goals))]

    for beta in beta_values:
        for user_id in range(n_synth_users_per_beta):
            state = graphManager.node2index[tuple(maps_filtered[str(level)]["start"])]
            orientation = 0
            x,y = graphManager.index2node[state]
            traj = [[x,y, orientation]]
            
            goal_to_reach = 0
            spAgent = deepcopy(spAgents[goal_to_reach])
            while len(traj) < max_length:
                action, actions, ps = spAgent.get_action(state, orientation, beta)
                state, orientation = spAgent.move(state, orientation, action)
                x,y = graphManager.index2node[state]
                traj += [[x, y, orientation]]
                
                if state == graphManager.goals[goal_to_reach]:
                    if goal_to_reach == len(graphManager.goals)-1:
                        break
                    else:
                        goal_to_reach += 1
                        spAgent = deepcopy(spAgents[goal_to_reach])
                    
            len_traj = len(traj)
            traj = np.array(traj).T
            path = {"x": traj[0], "y": traj[1], "r": traj[2]}
            row = [user_id, level, beta, len_traj, len_traj, path]
            synth_DF.append(row)
        print(f"Level: {level}, beta: {beta} completed")
        
    #if folder synth does not exist, create it
    if not os.path.exists("./synth"):
        os.makedirs("./synth")
        
    pd.DataFrame(synth_DF, columns=["user_id", "level", "beta", 
                                    "length", "length_interpolated", "path"]).to_csv(f"./synth/synth_ll_level_{level}.csv")
    
    
    print(f"Level: {level} completed")
    