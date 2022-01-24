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
from plot_funcs import plot_IO, miss_rate
from torch.utils.data import Dataset, DataLoader
import glob, os, re, sys, importlib
from check_data_quality import check_data_quality
from models import *
import matplotlib.tri as tri
from split_merge import split_grain, merge_grain
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


#==============================


grain_size = 2.5

evolve_runs = 1000 #num_test
frac_out = np.zeros((evolve_runs,frames,G)) ## final output
dy_out = np.zeros((evolve_runs,frames))
darea_out = np.zeros((evolve_runs,frames,G))


param_dat = np.zeros((evolve_runs, 2*G+4))
seq_1 = np.zeros((evolve_runs,1,3*G+1))

param_dat[:,G] = 2*float(sys.argv[1])
param_dat[:,G+1] = 1 - np.log10(float(sys.argv[2]))/np.log10(100) 
param_dat[:,G+2] = float(sys.argv[3])

## sample orientation

param_dat[:,G:2*G] = np.random.uniform(-1,1, evolve_runs*G).reshape((evolve_runs,G))

## sample frac 

frac = grain_size + 0.35*grain_size*np.random.randn(evolve_runs, G)
frac = frac/(G*grain_size)
fsum = np.cumsum(frac, axis=-1)
frac_change = np.diff((fsum>1)*(fsum-1),axis=-1,prepend=0) 
frac -= frac_change  
frac[:,-1] = np.ones(evolve_runs) - np.sum(frac[:,:-1], axis=-1)


print('sample frac', frac[0,:])
print('sample param', param_data[0,:])
assert np.linalg.norm( np.sum(frac, axis=-1) - np.ones(evolve_runs) ) <1e-5

param_dat[:,:G] = frac
seq_1[:,0,:G] = frac


#============================


      #  model


#==============================




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


      #  evolve


#==============================


alone = pred_frames%out_win
pack = pred_frames-alone



param_dat[:,-1] = dt
output_model = ini_model(todevice(seq_1), todevice(param_dat) )
dfrac_new = tohost( output_model[0] ) 
frac_new = tohost(output_model[1])
seq_dat = np.concatenate((seq_1,np.concatenate((frac_new, dfrac_new), axis = -1)),axis=1)


## write initial windowed data to out arrays
frac_out[:,:window,:] = seq_dat[:,:,:G]
dy_out[:,:window] = seq_dat[:,:,-1]
darea_out[:,:window,:] = seq_dat[:,:,2*G:3*G]

param_dat, seq_dat, expand = split_grain(param_dat, seq_dat, G_small, G)
print('the sub simulations', expand)

for i in range(0,pred_frames,out_win):
    
    param_dat[:,-1] = (i+window)*dt ## the first output time
    print('nondim time', (i+window)*dt)
    output_model = model(todevice(seq_dat), todevice(param_dat) )
    dfrac_new = tohost( output_model[0] ) 
    frac_new = tohost(output_model[1])

    if i>=pack:
        frac_out[:,-alone:,:], dy_out[:,-alone:], darea_out[:,-alone:,:] \
        = merge_grain(frac_new[:,:alone,:], dfrac_new[:,:alone,-1], dfrac_new[:,:alone,G_small:2*G_small], G_small, G, expand, area_coeff)
    else: 

        frac_out[:,window+i:window+i+out_win,:], dy_out[:,window+i:window+i+out_win], darea_out[:,window+i:window+i+out_win,:] \
        = merge_grain(frac_new, dfrac_new[:,:,-1], dfrac_new[:,:,G_small:2*G_small], G_small, G, expand, area_coeff)
    
    seq_dat = np.concatenate((seq_dat[:,out_win:,:], np.concatenate((frac_new, dfrac_new), axis = -1) ),axis=1)


dy_out = dy_out*y_norm
dy_out[:,0] = 0
y_out = np.cumsum(dy_out,axis=-1)+y_all[num_train:num_train+evolve_runs,[0]]

area_out = darea_out*area_norm


sio.savemat('e'+sys.argv[1]+'G'+sys.argv[2]+'R'+sys.argv[3]+'.mat',{'frac_out':frac_out,'y_out':y_out,'area_out':area_out,'param':param_dat})



#============================


      #  plot to check


#==============================









