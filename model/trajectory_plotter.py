import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import networkx as nx
import json
from scipy.spatial.distance import pdist
from graph_manager import GraphManager
import pandas as pd
import os
import imageio
import shutil


def extract_trajectory(DF_row):
    full_path = json.loads(DF_row)
    return full_path["x"], full_path["y"]

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



if __name__ == '__main__':
    if not os.path.exists('./model/temp/'):
        os.makedirs('./model/temp/')
    images = []
    
    save_mov = True
        
    
    level = "43"
    with open('./data/data_processed/maps_filtered.json', 'r') as fp:
        maps_filtered = json.load(fp)
    graphManager = GraphManager(maps_filtered[level]["nodes"], 
                                maps_filtered[level]["edges"], 
                                maps_filtered[level]["goals"])
    map_matrix = graph_to_imshow(maps_filtered[level])
    landmark_xs, landmark_ys = get_landmarks(maps_filtered[level])
        
    DF = pd.read_csv('./data/data_processed/Full_Dataset_Interpolated.csv')
    DF = DF[DF['level_id'] == int(level)]
    
    XS, YS = [[]*1000], [[]*1000]
    for subject_i in range(30):
        xs,ys = extract_trajectory(DF.iloc[subject_i]['path'])
        if len(xs) > len(XS):
            for step_j in range(len(xs)-len(XS)):
                XS.append([])
                YS.append([])
        
        for step_j in range(len(xs)):
            XS[step_j].append(xs[step_j])
            YS[step_j].append(ys[step_j])
            
    plt.ion()
    for step_j in range(len(XS[:500])):
        plt.scatter(landmark_xs, landmark_ys, s=100, c='red',zorder=-5)
        plt.text(landmark_xs[0], landmark_ys[0], "S", fontsize=20, zorder=-5,
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        for i in range(1, len(landmark_xs)):
            plt.text(landmark_xs[i], landmark_ys[i], i, fontsize=20, zorder=-5,
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        plt.imshow(map_matrix, cmap='gray_r', zorder=-15, origin='lower')
        
        
        x_jiggles = np.random.rand(len(XS[step_j]))-0.5
        y_jiggles = np.random.rand(len(YS[step_j]))-0.5
        plt.scatter(XS[step_j]+x_jiggles, YS[step_j]+y_jiggles, s=30, 
                    c=range(len(XS[step_j])), cmap='jet', zorder=5)
        if step_j > 0:
            plt.scatter(XS[step_j-1], YS[step_j-1], s=15, 
                    c=range(len(XS[step_j-1])), cmap='jet', zorder=4)
        if step_j > 1:
            plt.scatter(XS[step_j-2], YS[step_j-2], s=5, 
                    c=range(len(XS[step_j-2])), cmap='jet', zorder=3)
            
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.draw()
        if save_mov:
            plt.savefig('./model/temp/temp.png')
            images.append(imageio.imread('./model/temp/temp.png'))
        plt.pause(0.01)
        plt.clf()
        
        print(step_j, end="\r")
        
    if save_mov:
        imageio.mimsave('./model/out/movie.mp4', images, format="mp4", fps=8)
        shutil.rmtree('./model/temp', ignore_errors=True)
    