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


def ROM_qois(nx,ny,dx,G,angle_id,tip_y,frac,Ni0,Nif,area0,areaf):

   # note here frac dim [G,frames]
   # step 1: count the initial and final no. grain


    piece_len = np.asarray(np.round(frac*nx),dtype=int)
    piece0 = piece_len[:,0]
    
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
        
   # step 3: attach every grain to the size list
    for g in range(G):
         
        if piece_len[g,0]>0: 
            area0[angle_id[g]].append(ar0[g])
            areaf[angle_id[g]].append(arf[g])

# plot a bar graph with initial and final grain information:
# (1) the number of grains active on the S-L interface: total/num_simulations
# (2) the average and standard deviation of the grain area: total/num_grains

batch = 2500
num_batch = 4
num_runs = batch*num_batch

frames = 27
G = 20
bars = 10

# targets
Ni0 = np.zeros(bars,dtype=int) 
Nif = np.zeros(bars,dtype=int)    
di0 = np.zeros(bars,dtype=int)
dif = np.zeros(bars,dtype=int)   

# create list of lists, both of them should have the same size 
# size of each interval is Ni0, for each sublist, do mean and std
area0 = [[]] * bars
areaf = [[]] * bars


## start with loading data

datasets = glob.glob('../ML_PF10_train1000_test100_Mt47024_grains8_\
                     frames25_anis*_G05.000_Rmax1.000_seed*_rank0.h5')

f0 = datasets[0]
x = np.asarray(f0['x_coordinates'])
y = np.asarray(f0['y_coordinates'])
xmin = x[1]; xmax = x[-2]
ymin = y[1]; ymax = y[-2]
print('xmin',xmin,'xmax',xmax,'ymin',ymin,'ymax',ymax)
dx = x[1]-x[0]
fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
print('nx,ny', nx,ny)


for batch_id in range(num_batch):
  fname =datasets[batch_id]; print(fname)
  f = h5py.File(str(fname), 'r')
  aseq_asse = np.asarray(f['sequence'])
  frac_asse = np.asarray(f['fractions'])
  tip_y_asse = np.asarray(f['y_t'])
  #number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
  #print(number_list[6])
  # compile all the datasets interleave
  for run in range(batch):
    aseq = aseq_asse[run*G:(run+1)*G]  # 1 to 10
    frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
    tip_y = tip_y_asse[run*frames:(run+1)*frames]
    
    Color = (aseq-1)  # in the range of 0-9
    #print('angle sequence', Color)
    ROM_qois(nx,ny,dx,G,Color,tip_y,frac,Ni0,Nif,area0,areaf)

Ni0/=num_runs
Nif/=num_runs
print('initial distribution of average number of grains',Ni0)
print('final istribution of average number of grains',Nif)

for i in range(bars):
    print('Initial # grains:',len(area0[i]),' average:',np.mean(np.asarray(area0[i])))
    print('IFinal # grains:',len(areaf[i]),' average:',np.mean(np.asarray(areaf[i])))



