import numpy as np
from pr2_utils import *
import scipy as sp
import matplotlib.pyplot as plt
from pr2_lidar_transform import *
from pr2_predict import *

#start time
start_time = tic()

# init MAP
res = 1                                                      #map resolution
im = np.zeros((int(1600/res) + 1, int(1600/res) + 1)) + 0.5    #initialize grey color
x_im = np.arange(-100,1500+res, res)                           #x-axis of image
y_im = np.arange(-100,1500+res, res)                           #y-axis of image
lambda_max = 100
lambda_min = -100
N = 100                                                             #no. of particles
N_threshold = round(0.2*N)
alpha = np.full((N,1), 1/N)                                         #initial weight of each particle (N,1)
skip = 5                                                            #Update after 5 predict steps
xy = np.zeros((N,3))                                                #particles at a given timestep
traj = np.zeros((lidar_data.shape[0],3))                            #trajectory of best particle
reckon_traj = np.zeros((lidar_data.shape[0],3))                     #trajectory of dead reckoning
best_particles = np.zeros((int(lidar_data.shape[0]/skip) + 1, 3))   #initialize best particle

#Visualize Lidar in Polar Coordinates
# show_lidar(0)    

for i in range(lidar_data.shape[0]):

    #call predict
    xy = predict(xy, i)                                             #predict step
    # xy = dead_reckoning(xy,i)                                     #for dead reckoning
 
    #Update after 5 predict steps
    best_corr = np.zeros((N,1))
    if i%skip == 0:
        #lidar_to_vehicle
        lidar_world = update(N, xy, i)
    
        #Map Correlation
        for j in range(N):

            vp = lidar_world[j]     #(2,var)
            xrange, yrange = np.arange(-1,1.5,0.5) + xy[j,0] , np.arange(-1,1.5,0.5) + xy[j,1]   #range of points
            corr_mat = mapCorrelation(im, x_im, y_im, vp, xrange, yrange)
            best_corr[j,:] = np.max(corr_mat)

        best_weights = np.multiply(alpha, best_corr)
        norm_weights = softmax(best_weights)
        alpha = norm_weights
        best_particle_idx = np.argmax(alpha)
        best_particle = xy[best_particle_idx,0:2]         #position of best particle (2,)
        best_particles[int(i/skip),0:2]  = best_particle[np.newaxis,:]      #saving best particle for updated timesteps
        best_particles[int(i/skip),2] = alpha[best_particle_idx]            

        #Ray Tracing
        for k in range(lidar_world.shape[2]):
            
            sx, sy = best_particle[0]/res, best_particle[1]/res
            ex, ey = lidar_world[best_particle_idx,0,k]/res, lidar_world[best_particle_idx,1,k]/res
            bres_out = 0
            bres_out = bresenham2D(sx, sy, ex, ey)
            bres_out = bres_out.astype(int)

            # log odds
            if ((im[-bres_out[1,0:-2] +200, bres_out[0,0:-2] +200]).any() < lambda_max):
                im[-bres_out[1,0:-2] +200, bres_out[0,0:-2] +200] -= 1.5*np.log(4)      #updating log-odds for lidar rays (80% confidence)
            if ((im[-bres_out[1,-1] +200, bres_out[0,-1] +200]) > lambda_min):
                im[-bres_out[1,-1] +200, bres_out[0,-1] +200] += np.log(4)

 
    traj_idx = np.argmax(alpha)
    traj[i,0:2]  = xy[traj_idx,0:2][np.newaxis,:]
    traj[i,2] = xy[traj_idx,2]                      #saving best particle trajectory for all timesteps

    # reckon_traj[i,0] = xy[0,0]                       #dead reckoning trajectory
    # reckon_traj[i,1] = xy[0,1]

    #resampling
    if ((1/np.sum(alpha**2)) < N_threshold):     #compare N_effective with N_threshold
        resampled_xy = np.zeros(xy.shape)
        resampled_alpha = np.zeros(alpha.shape)
        a = 1
        c = alpha[0]
        for l in range(N):
            U = np.random.uniform(0,1/N)
            B = U + l/N
            while B > c:
                a = a + 1
                c = c + alpha[a]

            resampled_alpha += 1/N
            resampled_xy[l,:] = xy[a,:]
        alpha = resampled_alpha
        xy = resampled_xy

im[(im>-3.5) & (im != 0.5)] = 0
im[(im<-3.5) & (im != 0.5)] = 1

end_time = toc(start_time, name="Operation")
# np.save('trajectory_noise_01r_N200_final.npy', traj)
# np.save('log_odds_01r_N200_final.npy', im)
plt.figure()
# plt.scatter((traj[:,0]/res)+200, -(traj[:,1]/res)+200, color = 'r', s = 0.1)
# plt.scatter((reckon_traj[:,0]/res)+200, (reckon_traj[:,1]/res)+200, color = 'r', s = 0.1)
plt.imshow(im, cmap='gray')
plt.show(block=True)