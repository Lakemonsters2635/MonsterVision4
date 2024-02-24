import math
import random
import time
import json
import numpy as np
from numpy.linalg import lstsq
import pypotree as tree # pip install pypotree
import os
import matplotlib.pyplot as plt
import cv2
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
import seaborn as sns
import pandas as pd

# @profile
def FIT_PLANE(pointCloud, pointCount, plotCloud=True):
    # Reshape points to be a vertical stack of X, Y, and Z values; points.shape (3, n_points)
    points = pointCloud
    points = pointCloud.transpose()
    print(points.shape)
    points = np.vstack(points)
    print(points.shape)
    # Generate point cloud to be viewed
    cloud_path = tree.generate_cloud_for_display(points)
    plotCloud = False
    tree.display_cloud(cloud_path)
    if plotCloud:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(points[:,0], points[:,1], points[:,2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    # bs = points[:,:] #debuging
    # with open("/home/pi/MonsterVision4/TestPlot.txt", "w") as f:
    #     f.write(f"{[bs]}")
    print(xs.shape)

    A = np.vstack([xs, ys, np.ones_like(xs)]).T
    b = np.matrix(zs).T

    # print(A.shape)
    # print(b.shape)

    coefficients, residuals, rank, sigma = lstsq(A, b, rcond=None)
    # mean = np.mean(zs) 
    # print("mean")
    # print(mean)
    # print(f"{zs[:3]} + {mean} = {zs + mean}")
    
    # json.dump({
    #     "data": pointCloud,
    #     "coefficients": coefficients, 
    #     "residuals": residuals,
    #     "rank": rank,
    #     "sigma": sigma
    # }, "/tmp/delme.json")

