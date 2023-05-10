from bisect import bisect_right
import numpy as np
from pr2_utils import read_data_from_csv

T_lv = np.array([[0.00130201, 0.796097, 0.605167, 0.8349], [0.999999, -0.000419027, -0.00160026, -0.0126869], [-0.00102038, 0.605169, -0.796097, 1.76416], [0, 0, 0, 1]])
T_inv = np.linalg.inv(T_lv)

timestamp_lid, lidar_data = read_data_from_csv('code/data/sensor_data/lidar.csv')
timestamp_enc, data_enc = read_data_from_csv('code/data/sensor_data/encoder.csv')

def update(N, xy, i):

#-----------lidar to vehicle----------------------------

    time_idx = bisect_right(timestamp_lid, timestamp_enc[i])

    angles = np.linspace(-5, 185, 286) / 180 * np.pi        #in radians
    angles = angles[:,np.newaxis]                           #(286,1)

    m = lidar_data.shape[0]
    n = lidar_data.shape[1]

    r = lidar_data[time_idx][:,np.newaxis]                            #(286,1)     #check i
    filter_idx, _ = np.where((r > 1) & (r < 60))
    r = r[filter_idx,:]                                         #(var,1)
    angles = angles[filter_idx,:]
    r_cos = np.multiply(r, np.cos(angles))                     #X component (var,1)
    r_sin = np.multiply(r, np.sin(angles))                     #Y component (var,1)
    r_zeros = np.zeros((r.shape[0],1))                                           #No Z component in 2D Lidar (var,1)
    r_ones = np.ones((r.shape[0],1))                                             #Homogenous coordinate system (var,1)
    lidar_stack = np.hstack([r_cos, r_sin, r_zeros, r_ones])           #(var,4)

    lidar_vehicle = (np.matmul(T_inv,lidar_stack.transpose()))    #(4,var)

#-----------lidar vehicle to world-----------------------

    x = xy[:,0].reshape(N,1)  #(N,1)
    y = xy[:,1].reshape(N,1)  #(N,1)
    theta = xy[:,2].reshape(N,1)   #(N,1)
    T_world = np.hstack([np.stack([np.cos(theta), -np.sin(theta), np.zeros((N,1)), x], axis=2), np.stack([np.sin(theta), np.cos(theta), np.zeros((N,1)), y], axis=2)])  #(N,2,4)
    lidar_world = (np.matmul(T_world, lidar_vehicle))    #(N,2,var)

    return lidar_world

#---------------------------------------------------------
#map correlation

