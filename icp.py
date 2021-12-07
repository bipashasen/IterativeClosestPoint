import math
import json
from copy import deepcopy as c
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

np.set_printoptions(suppress=True)

def perform_ICP(P, Q, P_asnumpy, Q_asnumpy, P_coordinate, Q_coordinate):
    '''
    Function to perform ICP between two pointclouds - P and Q. 
    This function uses two sets of numpy arrays - 
    1. Downsampled for ICP alignment. 
    2. Non-downsampled for full pointcloud visualization. To align, it uses the matrix obtained from step 1.
    
    Parameters:
    P, Q: Point clouds to align.
    P_asnumpy, Q_asnumpy: Numpy values of P and Q respectively.
    P_coordinate, Q_coordinate: The coordinate frames of P and Q respectively.
    
    Function Parameters:
    P_ICP, Q_ICP: Point cloud used for Open3D visualization.
    P_asnumpy_ICP, Q_asnumpy_ICP: Downsampled (for faster processing) numpy array corresponding to P_ICP, Q_ICP. 
    P_asnumpy_visualize_ICP, Q_asnumpy_visualize_ICP: Non-Downsampled numpy values for visualization. 
    P_coordinate_ICP, Q_coordinate_ICP: The coordinate frames corresponding to P and Q.
    num_points_ICP, num_points: number of points in the downsampled pointclouds, non-downsampled pointclouds
    '''
        
    '''
    Initializations
    '''
    iterations = 80
    tolerance = 1e-5
    visualize_every = 8
    downsample_by = 50
    mean_distance = 9999
    tolerance_iteration = None

    '''
    P_ICP, Q_ICP:  Downsample P, Q points for faster execution.
    '''
    num_points_ICP = math.ceil(num_points/downsample_by)

    P_asnumpy_ICP = P_asnumpy[::downsample_by]

    Q_asnumpy_ICP = Q_asnumpy[::downsample_by]

    '''
    Using non-downsampled pointcloud for visualization.
    '''
    P_ICP, Q_ICP = c(P), c(Q)
    P_asnumpy_visualize_ICP, Q_asnumpy_visualize_ICP = c(P_asnumpy), c(Q_asnumpy)

    P_ICP.paint_uniform_color(ColorB)
    Q_ICP.paint_uniform_color(ColorG)

    P_coordinate_ICP, Q_coordinate_ICP = c(P_coordinate), c(Q_coordinate)

    print(f'Using {num_points_ICP} points for alignment.')

    '''
    Draw starting configuration.
    '''
    draw_geometry_pointcloud([P_ICP, P_coordinate_ICP, Q_ICP, Q_coordinate_ICP], saveas='icp-0', viewControlOptionJson='icp.json')

    '''
    ICP Implementation
    '''
    def find_nearest_neighbors(P_asnumpy, Q_asnumpy):
        '''
        P_asnumpy, Q_asnumpy as both n x 3.
        '''
        correspondence_indices, norms = [], []

        '''
        Distance = L2 distance between every pair: n x n 
        '''
        norms = np.linalg.norm(P_asnumpy_ICP[:, np.newaxis]-Q_asnumpy_ICP, axis=2)

        correspondence_indices = np.argmin(norms, axis=0)

        return Q_asnumpy[correspondence_indices]

    for i in range(iterations):
        current_mean_distance = np.linalg.norm(P_asnumpy_ICP-Q_asnumpy_ICP, axis=1).mean()

        distance_diff = mean_distance-current_mean_distance

        if distance_diff < tolerance and tolerance_iteration is None:
            tolerance_iteration = i

        mean_distance = current_mean_distance

        '''
        Find nearest neighbors
        '''
        Qtemp = find_nearest_neighbors(P_asnumpy_ICP, Q_asnumpy_ICP)

        '''
        Procrustes step
        '''
        R_ICP, t_ICP = procrustes_step(P_asnumpy_ICP.T, Qtemp.T)
        t_ICP = t_ICP.T[0]
        T_ICP = getTfromRt(R_ICP, t_ICP)

        '''
        Alignment
        '''
        P_asnumpy_ICP = np.stack([R_ICP@P_asnumpy_ICP[i].T + t_ICP for i in range(num_points_ICP)])
        P_asnumpy_visualize_ICP = np.stack([R_ICP@P_asnumpy_visualize_ICP[i].T + t_ICP for i in range(num_points)])

        P_ICP.points = o3d.utility.Vector3dVector(P_asnumpy_visualize_ICP)

        P_coordinate_ICP = P_coordinate_ICP.transform(T_ICP)

        if i%visualize_every == 0:
            print(f'Visualization for {i+1}(st/th/rd) step. Mean L2 Distance: {mean_distance}, Metric improvement: {distance_diff:.12f}')
            draw_geometry_pointcloud([P_ICP, P_coordinate_ICP, Q_ICP, Q_coordinate_ICP],
                saveas='icp-{}'.format(i+1), viewControlOptionJson='icp.json')

    print(f'Tolerance of {tolerance} reached at {tolerance_iteration}(st/th/rd) iteration')
    
'''
Example Call
perform_ICP(P, Q, P_asnumpy, Q_asnumpy, P_coordinate, Q_coordinate)
Parameters: 
Q, P: PointClouds
Q_asnumpy, P_asnumpy: Numpy form of the PointClouds Q and P respectively
Q_coordinate, P_coordinate: Coordinates of the pointclouds Q and P respectively.
'''
