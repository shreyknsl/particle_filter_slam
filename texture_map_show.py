import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from decimal import Decimal

traj = np.load('trajectory_noise_01.npy')
print('here2')
m1 = np.load('m1.npy')      #(1161,560,1280)
m2 = np.load('m2.npy')
m3 = np.load('m3.npy')
print('here3')

color_map = np.zeros((1500,1500,3))

folder = 'code/data/stereo_images/stereo_left'
img_no = 0

for filename in tqdm(sorted(os.listdir(folder))):
    image_l = cv2.imread(os.path.join(folder,filename),0)
    image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
    for i in range(560):
        for j in range(1280):
            x = m1[img_no,i,j]
            y = m2[img_no,i,j]
            z = m3[img_no,i,j]
            # print(x,y)
            if ((z<=0.5) & (z>=-0.5)):
                color_map[-int(y)+100, int(x)+100] = image_l[i,j]
    img_no += 1
    if img_no == 182:
        break

# plt.figure()
# plt.plot(traj[:,0]+100, -traj[:,1]+100, color='r', linewidth=1)
plt.imshow(color_map.astype(np.uint8))
plt.axis("off")
plt.show(block=True)