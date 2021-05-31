"""
        DBSCAN TEST FOR OUTLIER DETECTION
    
    Summary:
DBSCAN(Density-Based Spatial Clustering of Applications with Noise) is a commonly used unsupervised clustering algorithm proposed in 1996. 
Unlike the most well known K-mean, DBSCAN does not need to specify the number of clusters. It can automatically detect the number of clusters 
based on your input data and parameters. More importantly, DBSCAN can find arbitrary shape clusters that k-means are not able to find. 
For example, a cluster surrounded by a different cluster.

Also, DBSCAN can handle noise and outliers. All the outliers will be identified and marked without been classified into any cluster. 
Therefore, DBSCAN can also be used for Anomaly Detection (Outlier Detection)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import queue
import math

# CONSTANTS: Define label for different point group
NOISE = 0
UNASSIGNED = 0
core = -1
edge = -2

# Function to find all neigbor points in radius
def neighbor_points(data, pointId, radius):
    points = []
    for i in range(len(data)):
        # Euclidian distance using L2 Norm
        if np.linalg.norm(data[i] - data[pointId]) <= radius:
            points.append(i)
    return points

# DB Scan algorithom
def dbscan(data, Eps, MinPt, mode = 'scratch'):
    if mode == 'scratch':
        # Initialize all pointlable to unassign
        pointlabel  = [UNASSIGNED] * len(data)
        pointcount = []
        # Initilize list for core/noncore point
        corepoint = []
        noncore = []
        
        # Find all neigbor for all point
        for i in range(len(data)):
            pointcount.append(neighbor_points(train, i, Eps))
        
        # Find all core point, edgepoint and noise
        for i in range(len(pointcount)):
            if (len(pointcount[i]) >= MinPt):
                pointlabel[i]=core
                corepoint.append(i)
            else:
                noncore.append(i)

        for i in noncore:
            for j in pointcount[i]:
                if j in corepoint:
                    pointlabel[i]=edge

                    break
                
        # Start assigning point to luster
        cl = 1
        # Using a Queue to put all neigbor core point in queue and find neigboir's neigbor
        for i in range(len(pointlabel)):
            q = queue.Queue()
            if (pointlabel[i] == core):
                pointlabel[i] = cl
                for x in pointcount[i]:
                    if(pointlabel[x] == core):
                        q.put(x)
                        pointlabel[x] = cl
                    elif(pointlabel[x] == edge):
                        pointlabel[x] = cl
                #Stop when all point in Queue has been checked   
                while not q.empty():
                    neighbors = pointcount[q.get()]
                    for y in neighbors:
                        if (pointlabel[y] == core):
                            pointlabel[y] = cl
                            q.put(y)
                        if (pointlabel[y] == edge):
                            pointlabel[y] = cl            
                cl = cl + 1 # move to next cluster
    
    elif mode == 'library':
        data, Eps, MinPt
        pass

    return pointlabel, cl
    
# Function to plot final result
def plotRes(data, clusterRes, clusterNum, row, col, eps, minpts):
    nPoints = len(data)
    scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
    for i in range(clusterNum):
        if (i==0):
            #Plot all noise point as blue
            color='blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
    
        axs[row, col].scatter(x1, y1, c=color, alpha = 1, marker='.')
    
    axs[row, col].tick_params(labelsize = 5)
    axs[row, col].set_title('Clustering [eps = ' + str(eps) + ', MinPts = ' + str(minpts) + ']', fontsize = 7)
    axs[row, col].grid(color='black', alpha=0.05, linestyle='solid')


# Load Data
raw = pd.read_csv('DBSCAN.csv', delimiter =';')
train = raw[['x', 'y']].to_numpy()

# Set EPS and Minpoint
epss = [1, 2, 3]
minptss = [2, 4, 6]
fig, axs = plt.subplots(len(epss), len(minptss), constrained_layout = True)
# Find ALl cluster, outliers in different setting and print resultsw
for i, eps in enumerate(epss):
    for j, minpts in enumerate(minptss):
        print('Set eps = ' + str(eps) + ', Minpoints = '+ str(minpts))
        pointlabel, cl = dbscan(train, eps, minpts, mode = 'scratch')
        plotRes(train, pointlabel, cl, i, j, eps, minpts)
        print('number of cluster found: ' + str(cl-1))
        counter = collections.Counter(pointlabel)
        print(counter)
        outliers = pointlabel.count(0)
        print('numbrer of outliers found: ' + str(outliers) +'\n')

fig.suptitle('Parameter analysis')
plt.show()
