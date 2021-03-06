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
from video_plot import plot_synthetic
from torch.utils.data import Dataset, DataLoader
import glob, os, re, sys, importlib, copy
from check_data_quality import check_data_quality
from models import *
import matplotlib.tri as tri
from split_merge_reini import split_grain, merge_grain, assemb_seq, divide_seq
from scipy.interpolate import griddata
torch.cuda.empty_cache()

from G_E_test import *
mode = 'test'
noPDE = True
all_id = 42 #*out_case



frames_pool=[20,24,30]
learning_rate_pool=[25e-4, 50e-4, 100e-4]
layers_pool=[3,4,5]
hidden_dim_pool = [16, 24, 32]
out_win_pool = [3, 4, 5] 

frames_id = all_id//81
lr_id = (all_id%81)//27
layers_id = (all_id%27)//9
hd_id = (all_id%9)//3
owin_id = all_id%3

hidden_dim=hidden_dim_pool[hd_id]
learning_rate = learning_rate_pool[lr_id]
layers = layers_pool[layers_id]
frames = int(frames_pool[frames_id]*dilation)+1

out_win = out_win_pool[owin_id]


if mode=='train' or mode=='test':
  out_win+=1
  window=out_win

  train_frames=frames
  pred_frames= frames-window
if mode=='ini':
  learning_rate *= 2
  train_frames = window+out_win
  pred_frames = out_win
sam_per_run = frames - window - (out_win-1)
total_size = frames*num_runs
dt = dilation*1.0/(frames-1)

LSTM_layer = (layers, layers)
LSTM_layer_ini = (layers, layers)




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
evolve_runs = 1 #num_test

## sample orientation
np.random.seed(1)

f = h5py.File('ML_PF128_train0_test1_grains128_frames100_anis0.270_G04.750_Rmax1.820_seed1448122_rank0_grainsize2.500_Mt85200.h5', 'r')
w0 = np.asarray(f['fractions'])[:G_all]
#print(w0, np.where(w0==0),np.sum(w0))
#print('w0', w0, np.sum(w0))
angles_asse = np.asarray(f['angles'])
aseq_asse = np.asarray(f['sequence'])
#print(aseq_asse)
x = np.asarray(f['x_coordinates'])
y = np.asarray(f['y_coordinates'])
xmin = x[1]; xmax = x[-2]
ymin = y[1]; ymax = y[-2]
print('xmin',xmin,'xmax',xmax,'ymin',ymin,'ymax',ymax)
dx = x[1]-x[0]
fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;

ft = h5py.File('128grains100.h5', 'r')
alpha_asse = np.asarray(ft['alpha'])
alpha_true = alpha_asse[:fnx*fny].reshape((fnx,fny), order='F')[1:-1,1:-1]
alpha_true0 = alpha_true[:,0]


angle_all = np.zeros_like(angles_asse)
angle_all[1:] = angles_asse[1:]*180/pi + 90 

#print(angle_all,len(aseq_asse))

rand_angles = - ( 2*(angles_asse[aseq_asse] + pi/2)/(pi/2) - 1 )
rand_angles = rand_angles.reshape((evolve_runs,G_all))#np.random.uniform(-1,1, evolve_runs*G_all).reshape((evolve_runs,G_all))

true_count = np.zeros(G_all)

split_id = 0
doubles = 0


# taking an counter
dup = list(np.arange(G_all)+1)
# traversing the array
for item in alpha_true0:
    if item in dup:
        dup.remove(item)


print(dup)


for i in range(nx):

    if i>0 and alpha_true0[i]!=alpha_true0[i-1]:
        true_count[split_id] = i/nx
        split_id+=1
        
        while split_id+1 in dup: 
            true_count[split_id] = true_count[split_id-1]
            split_id+=1
        
print(split_id)
## sample frac 
print(doubles)
true_count[-1]=1
#print('frac',true_count)
true_count = np.concatenate((true_count[[0]],np.diff(true_count)))

frac = true_count.reshape((1,G_all))
print('frac',frac)

#frac = grain_size + std*grain_size*np.random.randn(evolve_runs, G_all)
#frac = frac/(G_all*grain_size)
'''
fsum = np.cumsum(frac, axis=-1)
frac_change = np.diff((fsum>1)*(fsum-1),axis=-1,prepend=0) 
frac -= frac_change  
frac[:,-1] = np.ones(evolve_runs) - np.sum(frac[:,:-1], axis=-1)
print(frac)
'''

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


if G_all>G:
    for i in range(batch_m):
        param_dat[i::batch_m,G:2*G] = rand_angles[:,i*G//2:i*G//2+G]
        bfrac = frac[:,i*G//2:i*G//2+G]/G*G_small
        off = -np.ones(bfrac.shape[0]) + np.sum(bfrac,axis=-1)
        if i==batch_m-1: bfrac= np.flip(bfrac, axis =-1)
        fsum = np.cumsum(bfrac, axis=-1)
        frac_change = np.diff((fsum>1)*(fsum-1),axis=-1,prepend=0) 
        bfrac -= frac_change  
        bfrac[:,-1] = np.ones(evolve_runs//batch_m) - np.sum(bfrac[:,:-1], axis=-1)
        if i==batch_m-1: bfrac= np.flip(bfrac, axis =-1)
        param_dat[i::batch_m,:G] = bfrac*G/G_small
        seq_1[i::batch_m,0,:G] = bfrac*G/G_small
        left_domain[i::batch_m] = np.sum(frac[:,:i*G//2], axis=-1)*G_small/G_all
        if i==batch_m-1: left_domain[i::batch_m] += off*G/G_all

else:
    param_dat[:,G:2*G] = rand_angles
    param_dat[:,:G] = frac
    seq_1[:,0,:G] = frac
  

#print('sample frac', seq_1[0,0,:])
#print('sample param', param_dat[0,:])
print('squence and param shape', seq_1.shape, param_dat.shape)
param_dat0 = param_dat
#============================


      #  model


#==============================

#frac_out[:,0,:] = seq_1[:,0,:G]
#dy_out[:,0] = seq_1[:,0,-1]
#darea_out[:,0,:] = seq_1[:,0,2*G:3*G]
#left_grains[:,0,:] = np.cumsum(frac_out[:,0,:], axis=-1) - frac_out[:,0,:]
seq_out[:,0,:] = seq_1[:,0,:]




def network_inf(seq_out,param_dat, model, ini_model, pred_frames, out_win, window):
    if noPDE == False:
        seq_dat = seq_test[:evolve_runs,:window,:]

        frac_out[:,:window,:] = seq_dat[:,:,:G]
        dy_out[:,:window] = seq_dat[:,:,-1]
        darea_out[:,:window,:] = seq_dat[:,:,2*G:3*G]

        param_dat, seq_dat, expand, left_coors = split_grain(param_dat, seq_dat, G_small, G)
    else: 


        seq_1 = seq_out[:,[0],:]   ## this can be generated randomly
        seq_1[:,:,-1]=0
        seq_1[:,:,G:2*G]=0
        print('sample', seq_1[0,0,:])

        param_dat_s, seq_1_s, expand, domain_factor, left_coors = split_grain(param_dat, seq_1, G_small, G)

        param_dat_s[:,-1] = dt
        domain_factor = size_scale*domain_factor
        seq_1_s[:,:,2*G_small:3*G_small] /= size_scale

        output_model = ini_model(todevice(seq_1_s), todevice(param_dat_s), todevice(domain_factor) )
        dfrac_new = tohost( output_model[0] ) 
        frac_new = tohost(output_model[1])

        dfrac_new[:,:,G_small:2*G_small] *= size_scale

        #frac_out[:,1:window,:], dy_out[:,1:window], darea_out[:,1:window,:], left_grains[:,1:window,:] \
        seq_out[:,1:window,:], left_grains[:,1:window,:] \
            = merge_grain(frac_new, dfrac_new, G_small, G, expand, domain_factor, left_coors)

        seq_dat = seq_out[:,:window,:]
        seq_dat_s = np.concatenate((seq_1_s,np.concatenate((frac_new, dfrac_new), axis = -1)),axis=1)
        if mode != 'ini':
          seq_dat[:,0,-1] = seq_dat[:,1,-1]
          seq_dat[:,0,G:2*G] = seq_dat[:,1,G:2*G] 
          seq_dat_s[:,0,-1] = seq_dat_s[:,1,-1]
          seq_dat_s[:,0,G:2*G] = seq_dat_s[:,1,G:2*G]
        #print(frac_new_vec.shape)

    ## write initial windowed data to out arrays

    #print('the sub simulations', expand)
    alone = pred_frames%out_win
    pack = pred_frames-alone

    for i in range(0,pred_frames,out_win):
        
        time_i = i
        if dt*(time_i+window+out_win-1)>1: 
            time_i = int(1/dt)-(window+out_win-1)
        ## you may resplit the grains here

        param_dat_s, seq_dat_s, expand, domain_factor, left_coors = split_grain(param_dat, seq_dat, G_small, G)

        param_dat_s[:,-1] = (time_i+window)*dt ## the first output time
        print('nondim time', (time_i+window)*dt)

        domain_factor = size_scale*domain_factor
        seq_dat_s[:,:,2*G_small:3*G_small] /= size_scale

        output_model = model(todevice(seq_dat_s), todevice(param_dat_s), todevice(domain_factor)  )
        dfrac_new = tohost( output_model[0] ) 
        frac_new = tohost(output_model[1])

        dfrac_new[:,:,G_small:2*G_small] *= size_scale

        if i>=pack and mode!='ini':
            seq_out[:,-alone:,:], left_grains[:,-alone:,:] \
            = merge_grain(frac_new[:,:alone,:], dfrac_new[:,:alone,:], G_small, G, expand, domain_factor, left_coors)
        else: 
            seq_out[:,window+i:window+i+out_win,:], left_grains[:,window+i:window+i+out_win,:] \
            = merge_grain(frac_new, dfrac_new, G_small, G, expand, domain_factor, left_coors)
        
        seq_dat = np.concatenate((seq_dat[:,out_win:,:], seq_out[:,window+i:window+i+out_win,:]),axis=1)
      

    frac_out, dfrac_out, darea_out, dy_out = divide_seq(seq_out, G)
    frac_out *= G_small/G
    dy_out = dy_out*y_norm
    dy_out[:,0] = 0
    y_out = np.cumsum(dy_out,axis=-1)+y0

    area_out = darea_out*area_norm
    return frac_out, y_out, area_out


def ensemble(seq_out, param_dat, inf_model_list):

    Nmodel = len(inf_model_list)


    frac_out = np.zeros((Nmodel,evolve_runs,frames,G)) ## final output
    area_out = np.zeros((Nmodel,evolve_runs,frames,G)) ## final output
    y_out = np.zeros((Nmodel,evolve_runs,frames))
    for i in range(Nmodel):

        seq_i = copy.deepcopy(seq_out)
        param_i = copy.deepcopy(param_dat)
        all_id = inf_model_list[i]
        frames_id = all_id//81
        lr_id = (all_id%81)//27
        layers_id = (all_id%27)//9
        hd_id = (all_id%9)//3
        owin_id = all_id%3

        hidden_dim=hidden_dim_pool[hd_id]
        layers = layers_pool[layers_id]

        out_win = out_win_pool[owin_id]
        out_win+=1
        window=out_win
        pred_frames= frames-window

        LSTM_layer = (layers, layers)
        LSTM_layer_ini = (layers, layers)

        model = ConvLSTM_seq(10, hidden_dim, LSTM_layer, G_small, out_win, kernel_size, True, device, dt)
        model = model.double()
        if device=='cuda':
            model.cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('total number of trained parameters ', pytorch_total_params)
        model.load_state_dict(torch.load('./fecr_model/lstmmodel'+str(all_id)))
        model.eval()  



        ini_model = ConvLSTM_start(10, hidden_dim, LSTM_layer_ini, G_small, window-1, kernel_size, True, device, dt)
        ini_model = ini_model.double()
        if device=='cuda':
           ini_model.cuda()
        init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
        print('total number of trained parameters for initialize model', init_total_params)
        ini_model.load_state_dict(torch.load('./fecr_model/ini_lstmmodel'+str(all_id)))
        ini_model.eval()


        frac_out[i,:,:,:], y_out[i,:,:], area_out[i,:,:,:] = network_inf(seq_i, param_i, model, ini_model, pred_frames, out_win, window)

  #  return frac_out/Nmodel, y_out/Nmodel, area_out/Nmodel  
    return np.mean(frac_out,axis=0), np.mean(y_out,axis=0), np.mean(area_out,axis=0)
#============================


      #  evolve input seq_1, param_dat


#==============================

## write initial windowed data to out arrays
inf_model_list = [42,24, 69,71]

#frac_out, y_out, area_out = ensemble(seq_out, param_dat, inf_model_list)




seq_reverse = copy.deepcopy(seq_out)
param_reverse = copy.deepcopy(param_dat)
seq_reverse [:,:,:G]      = np.flip(seq_out[:,:,:G],axis=-1)
seq_reverse [:,:,G:2*G]   = np.flip(seq_out[:,:,G:2*G],axis=-1)
seq_reverse [:,:,2*G:3*G] = np.flip(seq_out[:,:,2*G:3*G],axis=-1)

param_reverse [:,:G]      = np.flip(param_dat[:,:G],axis=-1)
param_reverse [:,G:2*G]   = -np.flip(param_dat[:,G:2*G],axis=-1)

print(seq_reverse[0,0,:],seq_out[0,0,:])
print(param_reverse[0,:],param_dat[0,:])

frac_out_r, y_out_r, area_out_r = ensemble(seq_reverse, param_reverse, inf_model_list)
frac_out_r= np.flip(frac_out_r,axis=-1)
area_out_r= np.flip(area_out_r,axis=-1)


frac_out_f, y_out_f, area_out_f = ensemble(seq_out, param_dat, inf_model_list)

frac_out = 0.5*(frac_out_f+frac_out_r)
y_out = 0.5*(y_out_f+y_out_r)
area_out = 0.5*(area_out_f+area_out_r)







sio.savemat('synthetic_grains'+str(G_all)+'_runs'+str(evolve_runs)+'_anis'+sys.argv[1]+'_G'+sys.argv[2]+'_Rmax'+sys.argv[3]+'.mat',{'frac':frac_out,'y':y_out,'area':area_out,'param':param_dat})



#============================


      #  plot to check


#==============================

nx_small = int(nx/G_all*G) 


pf_angles = np.zeros(G+1)
aseq_test = np.arange(G) +1

plot_flag = True
gap = 100//(frames-1)
print(gap, 'gap')
if plot_flag==True:
 plot_id = 0
 for dat_frames in [100]:
    data_id = np.arange(evolve_runs)[plot_id*batch_m:(plot_id+1)*batch_m] if G_all>G else [plot_id]
   # pf_angles[1:] = (param_dat0[data_id,G:2*G]+1)*45
    p_len = np.asarray(np.round(frac_out[data_id,:,:]*nx_small),dtype=int)
    left_grains = np.asarray(np.round(left_domain[data_id]*nx),dtype=int)
    pf_angles = np.concatenate((np.zeros((len(data_id),1),dtype=int), 90-(param_dat0[data_id,G:2*G]+1)*45), axis=-1)


    ft = h5py.File('128grains'+str(dat_frames)+'.h5', 'r')
    alpha_asse = np.asarray(ft['alpha'])
    alpha_true = alpha_asse[:fnx*fny].reshape((fnx,fny), order='F')[1:-1,1:-1]
    inf_frames = dat_frames//gap + 1
    extra_time = dat_frames/gap - dat_frames//gap
    area = area_out[data_id,inf_frames-1,:]
    if extra_time>0: area += extra_time*(area_out[data_id,inf_frames,:]-area_out[data_id,inf_frames-1,:])
    plot_synthetic(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),G,x,y,aseq_test,y_out[data_id,:], p_len, extra_time, dat_frames, inf_frames, \
    pf_angles, area, left_grains, nx_small, angle_all[alpha_true])








