#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:09:05 2021

@author: yigongqin
"""
from math import pi
import numpy as np
import scipy.io as sio
import h5py
import glob,os,re
from scipy.interpolate import interp1d
from math import pi
import matplotlib.pyplot as plt 
from scipy import io as sio 
import sys
from scipy import stats
plt.rcParams.update({'font.size': 10})

eps = 1e-4

def ROM_qois(nx,ny,dx,G,angle_id,tip_y,frac,extra_area,total_area,tip_y_f,Ni0,Nif,area0,areaf,ap_list):

   # note here frac dim [G,frames]
   # step 1: count the initial and final no. grain


    piece_len = np.asarray(np.round(frac*nx),dtype=int)
    piece0 = piece_len[:,0]
    #print(piece0) 
    for g in range(G):
        
        if piece_len[g,0]>eps: Ni0[angle_id[g]] +=1
        
        if piece_len[g,-1]>eps: Nif[angle_id[g]] +=1

   # step 2: area computing for every grain

    ar0 = total_area[:, 0]
    arf = total_area[:,-1]
        
   # step 3: calculate aspect ratio
    apf = np.zeros(G)
      # find the last active location
    for g in range(G):
        
        if piece_len[g,0]>eps: 

            height = tip_y_f[g,-1]-1
            apf[g] = height/( arf[g]/height )
            
        
   # step 4: attach every grain to the size list
    for g in range(G):
         
        if piece_len[g,0]>eps:

            area0.extend(list([ar0[g]]))
            areaf.extend(list([arf[g]]))
            ap_list.extend(list( [apf[g]]))
        #print(area0[angle_id[g]])
        #print(areaf[angle_id[g]])
# plot a bar graph with initial and final grain information:
# (1) the number of grains active on the S-L interface: total/num_simulations
# (2) the average and standard deviation of the grain area: total/num_grains

batch = 1000#2500
num_batch = 8
num_runs = batch*num_batch

frames = 24 +1
G = 8
bars = 100
# targets


## start with loading data
#datasets = glob.glob('../../mulbatch_train/*0.130*.h5')
#datasets = glob.glob('../../t_ML_PF10_train1000_test100_Mt23274_grains8_frames25_anis*.h5')
datasets = sorted(glob.glob('*.h5'))
#datasets = glob.glob('../../*test250_Mt70536*.h5')
#datasets = glob.glob('../../ML_PF10_train2250_test250_Mt70536_grains20_\
#                     frames27_anis0.130_G05.000_Rmax1.000_seed*_rank0.h5')
print(datasets)
f0 = h5py.File(str(datasets[0]), 'r')
x = np.asarray(f0['x_coordinates'])
y = np.asarray(f0['y_coordinates'])
xmin = x[1]; xmax = x[-2]
ymin = y[1]; ymax = y[-2]
print('xmin',xmin,'xmax',xmax,'ymin',ymin,'ymax',ymax)
dx = x[1]-x[0]
fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
print('nx,ny', nx,ny)

Ni0 = np.zeros(bars,dtype=int) 
Nif = np.zeros(bars,dtype=int)    
di0 = np.zeros((num_batch,bars))
dif = np.zeros((num_batch,bars))   
di0_std = np.zeros((num_batch,bars))
dif_std = np.zeros((num_batch,bars))    
aprf = np.zeros((num_batch,bars)) 

G0 = np.zeros(num_batch)
anis = np.zeros(num_batch)
Rmax = np.zeros(num_batch)

batchs = np.zeros(num_batch,dtype=int)

bins = np.arange(bars+1)

for batch_id in range(num_batch):
    fname =datasets[batch_id]; print(fname)
    f = h5py.File(str(fname), 'r')
    number_list=re.findall(r"[-+]?\d*\.\d+|\d+", fname)
    Rmax[batch_id] = number_list[8]
    G0[batch_id] = number_list[7]
    anis[batch_id] = number_list[6]
    aseq_asse = np.asarray(f['angles'])
    frac_asse = np.asarray(f['fractions'])
    tip_y_asse = np.asarray(f['y_t'])
    tip_y_f_asse = np.asarray(f['tip_y_f'])  
    extra_area_asse = np.asarray(f['extra_area'])
    total_area_asse = np.asarray(f['total_area'])   
  #angles_asse = np.asarray(f['angles'])
  #number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
  #print(number_list[6])
  # compile all the datasets interleave

    
    # create list of lists, both of them should have the same size 
    # size of each interval is Ni0, for each sublist, do mean and std
    area0 = []
    areaf = []
    
    apr_list = [] 
    batch = 1000
    for run in range(batch):
     # for run in range(1):
        aseq = aseq_asse[run*(G+1):(run+1)*(G+1)][1:]  # 1 to 10
        aseq = (aseq+pi/2)*180/pi
        frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
        tip_y = tip_y_asse[run*frames:(run+1)*frames]

        interval = np.asarray(aseq/10,dtype=int)
        assert np.all(interval>=0) and np.all(interval<9)
        extra_area = (extra_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
        total_area = (total_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
        tip_y_f    = (tip_y_f_asse   [run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')        
        #print('angle sequence', interval)
        ROM_qois(nx,ny,dx,G,interval,tip_y,frac,extra_area,total_area,tip_y_f,Ni0,Nif,area0,areaf,apr_list)

    di_f = np.sqrt(4.0*np.asarray(areaf)/pi)*dx 
    dif_g, _ = np.histogram(di_f , bins, density=True)
    apr_g, _ = np.histogram(apr_list, bins, density=True)
    exp_d = np.sum( (bins[1:]-0.5)*dif_g )
    exp_a = np.sum( (bins[1:]-0.5)*apr_g )

    converged = False
    batch = 10
    while not converged and batch<=500:
    ## ks test smaller than 5% and expectation smaller than 5%
      area0 = []
      areaf = []
    
      apr_list = [] 
      Ni0 = np.zeros(bars,dtype=int) 
      Nif = np.zeros(bars,dtype=int)   
      batch *= 2
      for run in range(batch):
     # for run in range(1):
        aseq = aseq_asse[run*(G+1):(run+1)*(G+1)][1:]  # 1 to 10
        aseq = (aseq+pi/2)*180/pi
        frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
        tip_y = tip_y_asse[run*frames:(run+1)*frames]

        interval = np.asarray(aseq/10,dtype=int)
        assert np.all(interval>=0) and np.all(interval<9)
        extra_area = (extra_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
        total_area = (total_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
        tip_y_f    = (tip_y_f_asse   [run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')        
        #print('angle sequence', interval)
        ROM_qois(nx,ny,dx,G,interval,tip_y,frac,extra_area,total_area,tip_y_f,Ni0,Nif,area0,areaf,apr_list)

        di_f = np.sqrt(4.0*np.asarray(areaf)/pi)*dx 
        dif_c, _ = np.histogram(di_f , bins, density=True)
        apr_c, _ = np.histogram(apr_list, bins, density=True)
        expectation_d = np.sum( (bins[1:]-0.5)*dif_c )
        expectation_a = np.sum( (bins[1:]-0.5)*apr_c )

        err_d = np.absolute(exp_d - expectation_d )/exp_d
        err_a = np.absolute(exp_a - expectation_a )/exp_a

        stats_1 = stats.kstest(dif_g, dif_c)[0]
        stats_2 = stats.kstest(apr_g, apr_c)[0]

        if stats_1<0.05 and stats_2<0.1 and err_d<0.01 and err_a<0.1: converged = True

    dif[batch_id,:] = dif_c
    aprf[batch_id,:] = apr_c
    print(batch, 'for ', batch_id)
    batchs[batch_id] = batch

print(G0)
print(anis)    
print(Rmax) 

fig,ax = plt.subplots(1,2,figsize=(15,5))

for i in range(num_batch):
    expectation = np.sum( (bins[1:]-0.5)*dif[i,:] )
    label='runs'+str(batchs[i])+'_G'+str(G0[i])+'_R'+str(Rmax[i])+'_e'+str(anis[i])+'_mean'+str(round(expectation, 3))
    if Rmax[i] > 0.5: ax[0].plot(dif[i,:], '--', label=label)
    else: ax[0].plot(dif[i,:], '-', label=label)
ax[0].set_xlim(0,30)
ax[0].set_xlabel(r'$d\ (\mu m)$')
ax[0].set_ylabel(r'$P$')
ax[0].legend()


for i in range(num_batch):
    expectation = np.sum( (bins[1:]-0.5)*aprf[i,:] )
    label='runs'+str(batchs[i])+'_G'+str(G0[i])+'_R'+str(Rmax[i])+'_e'+str(anis[i])+'_mean'+str(round(expectation, 3))
    if Rmax[i] > 0.5: ax[1].plot(aprf[i,:], '--', label=label)
    else: ax[1].plot(aprf[i,:], '-', label=label)
ax[1].set_xlim(0,40)    
ax[1].set_xlabel(r'$Asp$')
ax[1].set_ylabel(r'$P$')
ax[1].legend()


