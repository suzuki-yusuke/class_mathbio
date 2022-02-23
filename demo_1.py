# generate artificial cell activities


#%% import libs
import cv2
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm




#%% set parameters
T = 1000 # movie time
W = 20 # movie width
H = 20 # movie height

N = 1 # no. of cells
rad = 5 # cell radius
event_duration = 100
noise_level = 10
Lambda = 0.01 # parameter of Poisson process

num_bins = 20 # no. of bins for both axes
dst_image_size = 100

threshold_unit = 6




#%% generate artificial cell activities
spikes = np.random.poisson(Lambda, (N,T))
t_spike = [t for t in range(T) if spikes[0,t]==1] # when spiked
v_event = stats.gamma.pdf(np.arange(event_duration),2,scale=10) # response trace
v_threshold = stats.median_absolute_deviation(v_event,scale=1)*threshold_unit
v_true = np.zeros((1,T)).flatten() # true tarce
for t in t_spike:
    if (t+event_duration)<=T:
        v_true[t:t+event_duration] = v_true[t:t+event_duration] + v_event
    else:
        v_true[t:T] = v_true[t:T] + v_event[0:(T-t)]

v_true = v_true/v_threshold # normalize cell activities




#%% visualize artificial cell activities
#pbar = tqdm(total=T)
v_obs = np.zeros((1,T)).flatten()
for t in range(T):

    # generate signal image
    x = np.random.normal(0,rad,(2,1000))
    bin = np.linspace(-3*rad,rad*3,num_bins)
    h = np.histogram2d(x[0],x[1], bins=[bin,bin])
    cell_image = h[0]
    cv2.normalize(cell_image, cell_image, 0, 255, cv2.NORM_MINMAX)
    cell_image = cell_image*v_true[t]

    # generate background noise
    x = np.random.uniform(0,1,(2,1000))
    bin = np.linspace(0,1,num_bins)
    h = np.histogram2d(x[0],x[1], bins=[bin,bin])
    background_noise = h[0]*noise_level

    # generate observed image
    obs = cell_image + background_noise
    obs[np.isnan(obs)] = 0
    obs[obs<0] = 0
    obs[obs>255] = 255
    obs = np.uint8(obs)

    filtered = cv2.medianBlur(obs,3) # median filter

    # visualize
    obs = cv2.resize(obs, (int(dst_image_size), int(dst_image_size)))
    cv2.imshow('raw',obs)
    cv2.moveWindow('raw', 100, 200)

    filtered = cv2.resize(filtered, (int(dst_image_size), int(dst_image_size)))
    cv2.imshow('filtered',filtered)
    cv2.moveWindow('filtered', 300, 200)
    cv2.waitKey(1)

    v_obs[t] = np.nanmean(obs) # get mean pixell intensity

    #pbar.update(1)
    time.sleep(0.01)


cv2.destroyAllWindows()

v_obs = v_obs-np.min(v_obs)




#%% plot trace
fig, ax1 = plt.subplots(figsize=(8.0, 4.0))
cmap = plt.get_cmap("Dark2")

ax1.set_xlabel('t')
ax1.set_ylabel('true', color='r')
ax1.plot(v_true,'r-',linewidth=4)
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()
ax2.set_ylabel('observed', color='b')
ax2.plot(v_obs,'b-',linewidth=2)
ax2.tick_params(axis='y', labelcolor='b')






#%% make movie
I = np.zeros((H,W,T)) # movie
