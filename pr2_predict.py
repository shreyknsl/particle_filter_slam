import numpy as np
from pr2_utils import read_data_from_csv

timestamp_enc, data_enc = read_data_from_csv('code/data/sensor_data/encoder.csv')
timestamp_fog, data_fog = read_data_from_csv('code/data/sensor_data/fog.csv')


m_enc = data_enc.shape[0]

N = 100

# noise = np.random.randn(N, m_enc, 3)
# noise = np.random.normal(0,0.01, (N, m_enc, 3))

# encoder parameter
Dia_l = 0.623479      #m
Dia_r = 0.622806
res = 4096          #ticks/rev
meters_tick_l = np.pi * Dia_l / res     #arithmetic
meters_tick_r = np.pi * Dia_r / res

delta_enc = data_enc[1:m_enc] - data_enc[0:m_enc-1]
dr = delta_enc[:,1] * meters_tick_r
dl = delta_enc[:,0] * meters_tick_l
delta_enc = (0.5 * (dl + dr))[:,np.newaxis]
N_delta_enc = np.zeros((m_enc-1, N))
N_delta_enc = delta_enc
dead_reckon_enc = delta_enc                                         #dead reckoning
noise_enc = np.random.normal(0,0.01*np.abs(N_delta_enc),N_delta_enc.shape)
N_delta_enc = N_delta_enc + noise_enc                     #116047,N
# print(noisy_enc.shape)

yaw = data_fog[:,2]
yaw_sum10 = np.cumsum(yaw)
yaw_sum10 = yaw_sum10[10::10]
yaw_sum10 = (yaw_sum10[0:N_delta_enc.shape[0]])[:,np.newaxis]
N_yaw = np.zeros((yaw_sum10.shape[0], N))
N_yaw = yaw_sum10
dead_reckon_yaw = yaw_sum10                                         #dead reckoning
noise_yaw = np.random.normal(0,0.01*np.abs(N_yaw),N_yaw.shape)
N_yaw = N_yaw + noise_yaw

yaw_sin = np.sin(N_yaw)
yaw_cos = np.cos(N_yaw)

traj_x = N_delta_enc * yaw_cos
traj_y = N_delta_enc * yaw_sin

state_var = np.zeros((N_delta_enc.shape[0], N, 3))
state_var[:,:,0] = traj_x
state_var[:,:,1] = traj_y
state_var[:,:,2] = N_yaw

#dead reckoning
dead_reckon_x = dead_reckon_enc * np.cos(dead_reckon_yaw)
dead_reckon_y = dead_reckon_enc * np.sin(dead_reckon_yaw)

dead_reckon_var = np.zeros((N_delta_enc.shape[0], N, 3))
dead_reckon_var[:,0,0] = dead_reckon_x[:,0]
dead_reckon_var[:,0,1] = dead_reckon_y[:,0]
dead_reckon_var[:,0,2] = dead_reckon_yaw[:,0]

pos_predict = np.zeros((N,3))
def predict(xy, i):
    
    pos_predict[:,0] = state_var[i,:,0] + xy[:,0]
    pos_predict[:,1] = state_var[i,:,1] + xy[:,1]
    pos_predict[:,2] = state_var[i,:,2]

    return pos_predict

#dead reckoning
reckon = np.zeros((N,3))
def dead_reckoning(xy, i):
    reckon[0,0] = dead_reckon_var[i,0,0] + xy[0,0]
    reckon[0,1] = dead_reckon_var[i,0,1] + xy[0,1]
    reckon[0,2] = dead_reckon_var[i,0,2]
    return reckon