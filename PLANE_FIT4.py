import math
import random
import time
import numpy as np
from numpy.linalg import lstsq
import pypotree as tree # pip install pypotree
import os
import matplotlib.pyplot as plt
import cv2

# @profile
def FIT_PLANE(pointCloud, pointCount, plotCloud=True):
    # Reshape points to be a vertical stack of X, Y, and Z values; points.shape (3, n_points)
    points = pointCloud
    # points = pointCloud.transpose()
    # print(points.shape)
    # points = np.vstack(points)
    # print(points.shape)
    # Generate point cloud to be viewed
    # cloud_path = tree.generate_cloud_for_display(points)
    # tree.display_cloud(cloud_path)
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

    # print(xs.shape)

    A = np.vstack([xs, ys, np.ones_like(xs)]).T
    b = np.matrix(zs).T

    # print(A.shape)
    # print(b.shape)

    coefficients, residuals, rank, sigma = lstsq(A, b, rcond=None)