#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""
from math import pi
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_funcs import plot_synthetic
from torch.utils.data import Dataset, DataLoader
import glob, os, re, sys, importlib
from check_data_quality import check_data_quality
from models import *
import matplotlib.tri as tri
from split_merge_reini import split_grain, merge_grain, assemb_seq, divide_seq
from scipy.interpolate import griddata
torch.cuda.empty_cache()

from G_E_test import *

host='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device=host
print('device',device)

param_list = ['anis','G0','Rmax']

def todevice(data):
    return torch.from_numpy(data).to(device)
def tohost(data):
    return data.detach().to(host).numpy()

#============================


      #  synthetic
      #  dfrac, area, dy = 0 at t = 0 
      #  frac sample from normal distribution
      #  angle sample from unifrom distribution
      #  y0 = 2.25279999 because of the interface width


#==============================
#G_test = G

grain_size = 2.5
std = 0.35
y0 = 2.25279999
G_all = 128
evolve_runs = 3 #num_test

## sample orientation
np.random.seed(0)

rand_angles = np.random.uniform(-1,1, evolve_runs*G_all).reshape((evolve_runs,G_all))


## sample frac 

frac = grain_size + std*grain_size*np.random.randn(evolve_runs, G_all)
frac = frac/(G_all*grain_size)
fsum = np.cumsum(frac, axis=-1)
frac_change = np.diff((fsum>1)*(fsum-1),axis=-1,prepend=0) 
frac -= frac_change  
frac[:,-1] = np.ones(evolve_runs) - np.sum(frac[:,:-1], axis=-1)



assert np.linalg.norm( np.sum(frac, axis=-1) - np.ones(evolve_runs) ) <1e-5

frac = frac*G_all/G_small

batch_m = 1
if G_all>G:
    batch_m = (G_all-G//2)//(G//2)
    evolve_runs *= batch_m
#frac_out = np.zeros((evolve_runs,frames,G)) ## final output
#dy_out = np.zeros((evolve_runs,frames))
#darea_out = np.zeros((evolve_runs,frames,G))
seq_out = np.zeros((evolve_runs,frames,3*G+1))
left_grains = np.zeros((evolve_runs,frames,G))
left_domain = np.zeros((evolve_runs))

param_dat = np.zeros((evolve_runs, 2*G+4))
seq_1 = np.zeros((evolve_runs,1,3*G+1))

param_dat[:,2*G] = 2*float(sys.argv[1])
param_dat[:,2*G+1] = 1 - np.log10(float(sys.argv[2]))/np.log10(100) 
param_dat[:,2*G+2] = float(sys.argv[3])

for i in range(batch_m):
    param_dat[i::batch_m,G:2*G] = rand_angles[:,i*G//2:i*G//2+G]
    bfrac = frac[:,i*G//2:i*G//2+G]
    fsum = np.cumsum(bfrac, axis=-1)
    frac_change = np.diff((fsum>1)*(fsum-1),axis=-1,prepend=0) 
    bfrac -= frac_change  
    bfrac[:,-1] = np.ones(batch_m) - np.sum(bfrac[:,:-1], axis=-1)

    param_dat[i::batch_m,:G] = bfrac
    seq_1[i::batch_m,0,:G] = bfrac
    left_domain[i::batch_m] = np.sum(frac[:,:i*G//2], axis=-1)*G_small/G_all

print('sample frac', seq_1[0,0,:])
print('sample param', param_dat[0,:])
#============================


      #  model


#==============================

#frac_out[:,0,:] = seq_1[:,0,:G]
#dy_out[:,0] = seq_1[:,0,-1]
#darea_out[:,0,:] = seq_1[:,0,2*G:3*G]
#left_grains[:,0,:] = np.cumsum(frac_out[:,0,:], axis=-1) - frac_out[:,0,:]
seq_out[:,0,:] = seq_1[:,0,:]

model = ConvLSTM_seq(10, hidden_dim, LSTM_layer, G_small, out_win, kernel_size, True, device, dt)
model = model.double()
if device=='cuda':
  model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)

model.load_state_dict(torch.load('./lstmmodel0'))
model.eval()  

ini_model = ConvLSTM_start(10, hidden_dim, LSTM_layer_ini, G_small, window-1, kernel_size, True, device, dt)
ini_model = ini_model.double()
if device=='cuda':
   ini_model.cuda()
init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
print('total number of trained parameters for initialize model', init_total_params)
ini_model.load_state_dict(torch.load('./ini_lstmmodel0'))
ini_model.eval()



#============================


      #  evolve input seq_1, param_dat


#==============================

alone = pred_frames%out_win
pack = pred_frames-alone

param_dat0 = param_dat
param_dat_s, seq_1_s, expand, domain_factor, left_coors = split_grain(param_dat, seq_1, G_small, G)

param_dat_s[:,-1] = dt
domain_factor = size_scale*domain_factor
seq_1_s[:,:,2*G_small:3*G_small] /= size_scale

output_model = ini_model(todevice(seq_1_s), todevice(param_dat_s), todevice(domain_factor) )
dfrac_new = tohost( output_model[0] ) 
frac_new = tohost(output_model[1])

dfrac_new[:,:,G_small:2*G_small] *= size_scale

seq_out[:,1:window,:], left_grains[:,1:window,:] \
    = merge_grain(frac_new, dfrac_new, G_small, G, expand, domain_factor, left_coors)

seq_dat = seq_out[:,:window,:]
seq_dat[:,0,-1] = seq_dat[:,1,-1]
seq_dat[:,0,G:2*G] = seq_dat[:,1,G:2*G] 

## write initial windowed data to out arrays




#print('the sub simulations', expand)

for i in range(0,pred_frames,out_win):


    ## you may resplit the grains here

    param_dat_s, seq_dat_s, expand, domain_factor, left_coors = split_grain(param_dat, seq_dat, G_small, G)

    param_dat_s[:,-1] = (i+window)*dt ## the first output time
    print('nondim time', (i+window)*dt)

    domain_factor = size_scale*domain_factor
    seq_dat_s[:,:,2*G_small:3*G_small] /= size_scale

    output_model = model(todevice(seq_dat_s), todevice(param_dat_s), todevice(domain_factor)  )
    dfrac_new = tohost( output_model[0] ) 
    frac_new = tohost(output_model[1])

    dfrac_new[:,:,G_small:2*G_small] *= size_scale


    #if i>=pack:
     #   frac_out[:,-alone:,:], dy_out[:,-alone:], darea_out[:,-alone:,:], left_grains[:,-alone:,:] \
    #    = merge_grain(frac_new[:,:alone,:], dfrac_new[:,:alone,-1], dfrac_new[:,:alone,G_small:2*G_small], G_small, G, expand, domain_factor, left_coors)
   # else: 

   # frac_out[:,window+i:window+i+out_win,:], dy_out[:,window+i:window+i+out_win], darea_out[:,window+i:window+i+out_win,:], left_grains[:,window+i:window+i+out_win,:] \
    seq_out[:,window+i:window+i+out_win,:], left_grains[:,window+i:window+i+out_win,:] \
    = merge_grain(frac_new, dfrac_new, G_small, G, expand, domain_factor, left_coors)
    
    seq_dat = np.concatenate((seq_dat[:,out_win:,:], seq_out[:,window+i:window+i+out_win,:]),axis=1)

frac_out, dfrac_out, darea_out, dy_out = divide_seq(seq_out, G)
frac_out *= G_small/G
dy_out = dy_out*y_norm
dy_out[:,0] = 0
y_out = np.cumsum(dy_out,axis=-1)+y0

area_out = darea_out*area_norm

sio.savemat('synthetic'+str(evolve_runs)+'_anis'+sys.argv[1]+'_G'+sys.argv[2]+'_Rmax'+sys.argv[3]+'.mat',{'frac':frac_out,'y':y_out,'area':area_out,'param':param_dat})



#============================


      #  plot to check


#==============================
datasets = sorted(glob.glob(data_dir))
print('dataset dir',data_dir,' and size',len(datasets))
filename = datasets[0]
#filename = filebase+str(2)+ '_rank0.h5'
f = h5py.File(filename, 'r')
x = np.asarray(f['x_coordinates'])
y = np.asarray(f['y_coordinates'])

dx = x[1] - x[0]
nx = len(x) -3
nx_all = int(nx*G_all/G) + 1
x = np.linspace(0, x[-2]*G_all/G+dx, nx_all)

pf_angles = np.zeros(G+1)
aseq_test = np.arange(G) +1

for plot_id in range(3):
    data_id = np.arange(evolve_runs)[plot_id*batch_m:(plot_id+1)*batch_m] if G_all>G else plot_id
   # pf_angles[1:] = (param_dat0[data_id,G:2*G]+1)*45
    plot_synthetic(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),G,x,y,aseq_test,y_out[data_id,:][0],frac_out[data_id,:,:][0].T, \
        plot_id, train_frames, np.concatenate(([0],(param_dat0[data_id,G:2*G][0]+1)*45)), area_out[data_id,train_frames-1,:][0], left_domain[data_id])
















