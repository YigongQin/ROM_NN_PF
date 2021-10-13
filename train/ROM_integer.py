#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:09:05 2021

@author: yigongqin
"""

import numpy as np
import scipy.io as sio
import h5py
import glob,os,re
from scipy.interpolate import interp1d
from math import pi
import matplotlib.pyplot as plt 
from scipy import io as sio 
import sys

def ROM_qois(nx,ny,dx,G,angle_id,tip_y,frac,Ni0,Nif,area0,areaf,ap_list):

   # note here frac dim [G,frames]
   # step 1: count the initial and final no. grain


    piece_len = np.asarray(np.round(frac*nx),dtype=int)
    piece0 = piece_len[:,0]
    #print(piece0) 
    for g in range(G):
        
        if piece_len[g,0]>0: Ni0[angle_id[g]] +=1
        
        if piece_len[g,-1]>0: Nif[angle_id[g]] +=1

   # step 2: area computing for every grain

    ntip_y = np.asarray(tip_y/dx,dtype=int)    


    ar0 = piece0*(ntip_y[0]+1)
    arf = np.zeros(G, dtype=int)
    #temp_piece = np.zeros(G, dtype=int)
    yj = np.arange(ntip_y[0]+1,ntip_y[-1]+1)

    for g in range(G):
        fint = interp1d(ntip_y, piece_len[g,:],kind='linear')
        new_f = fint(yj)
        grid_piece = np.asarray(new_f,dtype=int)  
        # here remains the question whether it should be interger
        arf[g] = np.sum(grid_piece) + ar0[g]   # sum from 0-tip0, tip0+1 to tip-1
        
   # step 3: calculate aspect ratio
    apf = np.zeros(G)
      # find the last active location
    for g in range(G):
        
        if piece_len[g,0]>0: 
            if piece_len[g,-1]<0:
              height_id = list((piece_len[g,:]>0)*1).index(0)-1
            else: height_id = frames-1
            height = ntip_y[height_id]
            apf[g] = height/( arf[g]/height )
            
        
   # step 4: attach every grain to the size list
    for g in range(G):
         
        if piece_len[g,0]>0:
            list0g = area0[angle_id[g]]
            listfg = areaf[angle_id[g]]
            list0g.extend([ar0[g]])
            listfg.extend([arf[g]])
            listap = ap_list[angle_id[g]]
            listap.extend([apf[g]])
        #print(area0[angle_id[g]])
        #print(areaf[angle_id[g]])
# plot a bar graph with initial and final grain information:
# (1) the number of grains active on the S-L interface: total/num_simulations
# (2) the average and standard deviation of the grain area: total/num_grains

batch = 200#2500
num_batch = 4
num_runs = batch*num_batch

frames = 25 +1
G = 8
bars = 10
pfs=11
# targets


## start with loading data
#datasets = glob.glob('../../mulbatch_train/*0.130*.h5')
#datasets = glob.glob('../../t_ML_PF10_train1000_test100_Mt23274_grains8_frames25_anis*.h5')
datasets = glob.glob('*.h5')
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

for batch_id in range(num_batch):
    fname =datasets[batch_id]; print(fname)
    f = h5py.File(str(fname), 'r')
    number_list=re.findall(r"[-+]?\d*\.\d+|\d+", fname)
    G0[batch_id] = number_list[7]
    anis[batch_id] = number_list[6]
    aseq_asse = np.asarray(f['sequence'])
    frac_asse = np.asarray(f['fractions'])
    tip_y_asse = np.asarray(f['y_t'])
  #angles_asse = np.asarray(f['angles'])
  #number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
  #print(number_list[6])
  # compile all the datasets interleave

    
    # create list of lists, both of them should have the same size 
    # size of each interval is Ni0, for each sublist, do mean and std
    area0 = [[] for _ in range(bars)]
    areaf = [[] for _ in range(bars)]
    
    apr_list = [[] for _ in range(bars)]
    for run in range(batch):
     # for run in range(1):
        aseq = aseq_asse[run*G:(run+1)*G]  # 1 to 10
        frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
        tip_y = tip_y_asse[run*frames:(run+1)*frames]

        interval = aseq-1 #np.asarray(Color/10,dtype=int)
        #print('angle sequence', interval)
        ROM_qois(nx,ny,dx,G,interval,tip_y,frac,Ni0,Nif,area0,areaf,apr_list)

    Ni0=Ni0/batch
    Nif=Nif/batch
    #print('initial distribution of average number of grains',Ni0)
    #print('final istribution of average number of grains',Nif)
    #print(area0)
    
    for i in range(bars):
    #    print('Initial # grains:',len(area0[i]),' average:',dx**2*np.mean(np.asarray(area0[i])))
    #    print('Final # grains:',len(areaf[i]),' average:',dx**2*np.mean(np.asarray(areaf[i])))
       di0[batch_id,i] = np.mean( np.sqrt(4.0*np.asarray(area0[i])/pi) )*dx
       dif[batch_id,i] = np.mean( np.sqrt(4.0*np.asarray(areaf[i])/pi) )*dx
       di0_std[batch_id,i] = np.std( np.sqrt(4.0*np.asarray(area0[i])/pi) )*dx/np.sqrt(len(area0[i]))
       dif_std[batch_id,i] = np.std( np.sqrt(4.0*np.asarray(areaf[i])/pi) )*dx/np.sqrt(len(areaf[i]))
       aprf[batch_id,i] = np.mean(np.asarray(apr_list[i]))
    #print('initial distribution of average grain diameter',di0)
    #print('final istribution of average grain diameter',dif)
    #print(di0_std,dif_std)


print(G0)
print(anis)    
sio.savemat('interp.mat',{'G0':G0,'anis':anis,'dif':dif})
plt.plot(dif[0,:])


