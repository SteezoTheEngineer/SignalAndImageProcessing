"""

Introduction to Signal and Image Processing
Homework 1

Created on Mar 17, 2020
@author: Stefan Weber
    
"""

import glob
import os
import numpy as np
import math # for Chamfering Algorithm
import matplotlib.pyplot as plt

# distance array
al = np.array([2,1,1,2])

# load shapes
shapes = glob.glob(os.path.join('shapes', '*.png'))
for i, shape in enumerate(shapes):  
    # load the edge map
    edge_map = plt.imread(shape, format=None)

    # caclulate distance map
    # distance_map: array_like, same size as edge_map
    # copy from edge map
    distance_map = edge_map
    # set all not edges inf
    distance_map = np.where(distance_map == 0, np.inf, distance_map)
    # set all edges 0
    distance_map = np.where(distance_map == 1, 0, distance_map)
    # add frame that distance of all pixels can be caluclated
    distance_map = np.insert(distance_map, 0, np.inf, axis=1)
    distance_map = np.insert(distance_map, 0, np.inf, axis=0)
    distance_map = np.insert(distance_map, distance_map.shape[1], np.inf, axis=1)
    distance_map = np.insert(distance_map, distance_map.shape[0], np.inf, axis=0)


    # Distance with ul filter
    for x in range(1, distance_map.shape[0]-1):
        for y in range(1, distance_map.shape[1]-1):
            dist = np.array([np.inf, np.inf, np.inf, np.inf])
            dist[0] = distance_map[x-1,y-1] + al[0]
            dist[1] = distance_map[x-1,y] + al[1]
            dist[2] = distance_map[x,y-1] + al[2]
            dist[3] = distance_map[x+1,y-1] + al[3]
            
            if (np.amin(dist) < distance_map[x,y] and np.amin(dist) != np.inf):
                distance_map[x,y] = np.amin(dist) 

    # Distance with br filter
    for x in range(distance_map.shape[0]-2, 1, -1):
        for y in range(distance_map.shape[1]-2, 1, -1):
            dist = np.array([np.inf, np.inf, np.inf, np.inf])
            dist[0] = distance_map[x+1,y+1] + al[0]
            dist[1] = distance_map[x+1,y] + al[1]
            dist[2] = distance_map[x,y+1] + al[2]
            dist[3] = distance_map[x-1,y+1] + al[3]
            
            if (np.amin(dist) < distance_map[x,y] and np.amin(dist) != np.inf):
                distance_map[x,y] = np.amin(dist) 

    distance_map = distance_map[1:-1,1:-1]
    # the top row of the plots should be the edge maps, and on the bottom the corresponding distance maps
    k, l = i+1, i+len(shapes)+1
    plt.subplot(2, len(shapes), k)
    plt.imshow(edge_map, cmap='gray')
    plt.subplot(2, len(shapes), l)
    plt.imshow(distance_map, cmap='gray')
    #plt.savefig('Chamfer_Distance.jpg')

plt.show()
