import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from pr2_utils import compute_stereo, tic, toc, read_data_from_csv
from bisect import *
from decimal import *

#parameter--------------------

b = 0.475143600050775
K = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02],[0, 7.7537235550066748e+02, 2.5718049049377441e+02],[0., 0., 1.]])
fs_u, fs_v = K[0,0], K[1,1]
c_u, c_v = round(K[0,2]), round(K[1,2])
Rv = np.array([[-0.00680499, -0.0153215, 0.99985], [-0.999977, 0.000334627, -0.00680066], [-0.000230383, -0.999883, -0.0153234]])
p = np.array([[1.64239], [0.247401], [1.58411]])

u_l = np.zeros((560,1280))
u_l[:] = np.arange(0,1280,1)

v_l = np.zeros((1280,560))
v_l[:] = np.arange(0,560,1)
v_l = v_l.transpose()     #(560,1280)

x_opt = np.zeros((1161,560,1280))
y_opt = np.zeros((1161,560,1280))
z_opt = np.zeros((1161,560,1280))
d = np.zeros((1161,560,1280))
timestamp = np.zeros((1161,1))
# m = np.zeros((1161,560,1280,3))
m1 = np.zeros((1161,560,1280))
m2 = np.zeros((1161,560,1280))
m3 = np.zeros((1161,560,1280))
# img_world = np.zeros((1161,560,1280,4))

timestamp_lid, _ = read_data_from_csv('code/data/sensor_data/lidar.csv')
trajectory = np.load('trajectory_noise_01.npy')

x = trajectory[:,0]
y = trajectory[:,1]
theta = trajectory[:,2]
T = np.zeros((4,4))

path_l = 'code/data/stereo_images/stereo_left'
file_l = []
temp_l = 0
for filename_l in sorted(os.listdir(path_l)):
    temp_l, _ = filename_l.split('.',1)
    file_l = np.append(file_l,Decimal(temp_l))

path_r = 'code/data/stereo_images/stereo_right'
file_r = []
temp_r = 0
for filename_r in sorted(os.listdir(path_r)):
    temp_r, _ = filename_r.split('.',1)
    file_r = np.append(file_r,Decimal(temp_r))

file_rr = []
for j in file_r:
    idx = np.argmin(np.abs(file_l - j))
    file_rr = np.append(file_rr, file_r[idx])

i = 0
# for filename in tqdm(sorted(os.listdir(folder))):
for i in tqdm(range(file_l.shape[0])):

    img_l = str(file_l[i]) + '.png'
    img_r = str(file_rr[i]) + '.png'

    # convert to optical

    d[i] = compute_stereo(path_l, path_r, img_l, img_r)
    z_opt[i] = np.divide(fs_u*b, d[i])
    z_opt[i][z_opt[i]>50] = 0
    x_opt[i] = (u_l - c_u)*z_opt[i]/fs_u
    y_opt[i] = (v_l - c_v)*z_opt[i]/fs_v
    timestamp[i] = file_l[i]

    #convert to vehicle

    xyz_opt = np.stack([x_opt[i], y_opt[i], z_opt[i]], axis = 2)        #(3,1280)

    mul = np.zeros((560,1280,3))
    for j in range(560):
        mul[j] = (np.matmul(Rv, xyz_opt[j].transpose())).transpose()

    m1[i] = mul[:,:,0]
    m2[i] = mul[:,:,1]
    m3[i] = mul[:,:,2]

    #convert to world

    time_idx = bisect_right(timestamp_lid, timestamp[i])
    T = np.array([[np.cos(theta[time_idx]), -np.sin(theta[time_idx]), 0, x[time_idx]],[np.sin(theta[time_idx]), np.cos(theta[time_idx]), 0, y[time_idx]],[0,0,1,0],[0,0,0,1]]) #(4,4)
    m_homo = np.stack([m1[i], m2[i], m3[i], np.ones((560,1280))], axis = 2) #(560,1280,4)

    temp = np.zeros((560,1280,4))
    for j in range(560):
        temp[j] = np.matmul(T,m_homo[j].transpose()).transpose()

    m1[i] = temp[:,:,0]
    m2[i] = temp[:,:,1]
    m3[i] = temp[:,:,2]

    i+=1

np.save('m1', m1)
np.save('m2', m2)
np.save('m3', m3)
print('finish')