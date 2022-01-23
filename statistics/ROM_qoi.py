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
plt.rcParams.update({'font.size': 12})

eps = 1e-4


def true_qois(nx,ny,dx,G,angle_id,tip_y,frac,extra_area,total_area,tip_y_f,Ni0,Nif,area0,areaf,ap_list):

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

def ROM_qois(nx,ny,dx,G,angle_id,tip_y,frac,extra_area,Ni0,Nif,area0,areaf,ap_list):

   # note here frac dim [G,frames]
   # step 1: count the initial and final no. grain


    piece_len = np.asarray(np.round(frac*nx),dtype=int)
    piece0 = piece_len[:,0]
    #print(piece0) 
    for g in range(G):
        
        if piece_len[g,0]>eps: Ni0[angle_id[g]] +=1
        
        if piece_len[g,-1]>eps: Nif[angle_id[g]] +=1

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
        arf[g] = np.sum(grid_piece) + ar0[g] + extra_area[g,-1]  # sum from 0-tip0, tip0+1 to tip-1
        
   # step 3: calculate aspect ratio
    apf = np.zeros(G)
      # find the last active location
    for g in range(G):
        
        if piece_len[g,0]>eps: 
            if piece_len[g,-1]<eps:
              height_id = list((piece_len[g,:]>eps)*1).index(0)-1
              height = ntip_y[height_id]
            else: 
              height_id = frames-1
              if piece_len[g,-1]>eps:
                  height = ntip_y[height_id] + extra_area[g,-1]/piece_len[g,-1]
              else: height = ntip_y[height_id]
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
datasets = sorted(glob.glob('*PF8*154272*.h5'))
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
dit = np.zeros((num_batch,bars))     
di0_std = np.zeros((num_batch,bars))
dif_std = np.zeros((num_batch,bars))    
aprf = np.zeros((num_batch,bars)) 
aprt = np.zeros((num_batch,bars)) 

y_t = np.zeros((num_batch, frames))
df_t = np.zeros((num_batch, frames-1))
area_t = np.zeros((num_batch, frames))

G0 = np.zeros(num_batch)
anis = np.zeros(num_batch)
Rmax = np.zeros(num_batch)

batchs = np.zeros(num_batch,dtype=int)

bins = np.arange(bars+1)

ks_d = np.zeros(num_batch)
ks_a = np.zeros(num_batch)
err_d = np.zeros(num_batch)
err_a = np.zeros(num_batch)

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
    extra_area_asse = np.asarray(f['extra_area'])
    print('max of extra area',np.max(extra_area_asse))
    tip_y_f_asse = np.asarray(f['tip_y_f']) 
    total_area_asse = np.asarray(f['total_area'])
    y_t[batch_id,:] = np.mean(tip_y_asse[:frames*batch].reshape((batch,frames)), axis = 0)
    area_t[batch_id,:] = np.mean(np.mean( extra_area_asse[:frames*batch*G].reshape((batch,frames,G)), axis = 0), axis = -1)
    df_t[batch_id,:] = np.mean(np.mean( np.absolute(np.diff(frac_asse[:frames*batch*G].reshape((batch,frames,G)),axis=1 )) , axis = 0), axis = 1)
  #angles_asse = np.asarray(f['angles'])
  #number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
  #print(number_list[6])
  # compile all the datasets interleave

    
    # create list of lists, both of them should have the same size 
    # size of each interval is Ni0, for each sublist, do mean and std
    area0 = []
    areaf = []
    
    apr_list = []

    area0_t = []
    areaf_t = []
    
    apr_list_t = []   
    for run in range(batch):
     # for run in range(1):
        aseq = aseq_asse[run*(G+1):(run+1)*(G+1)][1:]  # 1 to 10
        aseq = (aseq+pi/2)*180/pi
        frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
        tip_y = tip_y_asse[run*frames:(run+1)*frames]

        interval = np.asarray(aseq/10,dtype=int)
        assert np.all(interval>=0) and np.all(interval<9)
        #print('angle sequence', interval)
        extra_area = (extra_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
        total_area = (total_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
        tip_y_f    = (tip_y_f_asse   [run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')        
        ROM_qois(nx,ny,dx,G,interval,tip_y,frac,extra_area,                    Ni0,Nif,area0,areaf,apr_list)
        true_qois(nx,ny,dx,G,interval,tip_y,frac,extra_area,total_area,tip_y_f,Ni0,Nif,area0_t,areaf_t,apr_list_t)
 
    di_f = np.sqrt(4.0*np.asarray(areaf)/pi)*dx 
    dif[batch_id,:], _ = np.histogram(di_f , bins, density=True)
    aprf[batch_id,:], _ = np.histogram(apr_list, bins, density=True)

    di_t = np.sqrt(4.0*np.asarray(areaf_t)/pi)*dx 
    dit[batch_id,:], _ = np.histogram(di_t , bins, density=True)
    aprt[batch_id,:], _ = np.histogram(apr_list_t, bins, density=True)

    batchs[batch_id] = batch

    ks_d[batch_id]= stats.kstest(dif[batch_id,:], dit[batch_id,:])[0]
    ks_a[batch_id]= stats.kstest(aprf[batch_id,:], aprt[batch_id,:])[0]

    expectation_d = np.sum( (bins[1:]-0.5)*dif[batch_id,:])
    expectation_a = np.sum( (bins[1:]-0.5)*aprf[batch_id,:] )
    exp_d = np.sum( (bins[1:]-0.5)*dit[batch_id,:] )
    exp_a = np.sum( (bins[1:]-0.5)*aprt[batch_id,:])
    err_d[batch_id] = np.absolute(exp_d - expectation_d )/exp_d
    err_a[batch_id] = np.absolute(exp_a - expectation_a )/exp_a

print(G0)
print(anis)   
print(Rmax) 


np.set_printoptions(precision=3)
print(ks_d)
print(ks_a)
print(err_d)
print(err_a)

line_styles = ['b-','r--','c-','g--','bs','rs-','cs','gs-']
t = np.linspace(0,90,25)

fig,ax = plt.subplots(1,3,figsize=(20,5))
case = ['a','b','c','d','e','f','g','h']
for i in range(num_batch):
 
    label=case[i]
    ax[0].plot(t,y_t[i,:], line_styles[i], label=label)
    ax[0].set_ylabel(r'$y\ (\mu m) $')
    ax[0].set_xlabel(r'$t\ (\mu s)$')
ax[0].legend()

for i in range(num_batch):
    label=case[i]
    ax[1].plot(t[1:],df_t[i,:], line_styles[i], label=label)
    ax[1].set_xlabel(r'$t\ (\mu s)$')
    ax[1].set_ylabel(r'$\bar{\Delta w} $')
ax[1].legend()

for i in range(num_batch):
    label=case[i]
    ax[2].plot(t, area_t[i,:]*dx**2, line_styles[i], label=label)
    ax[2].set_xlabel(r'$t\ (\mu s)$')
    ax[2].set_ylabel(r'$S (\mu m^2)$')
ax[2].legend()

fig.savefig('y_w.pdf',dpi=600)

fig,ax = plt.subplots(1,2,figsize=(15,5))

for i in range(num_batch):
    expectation = np.sum( (bins[1:]-0.5)*dif[i,:] )
    label=case[i]+': runs'+str(batchs[i])+'_mean'+str(round(expectation, 3))
    ax[0].plot(dif[i,:], line_styles[i], label=label)
ax[0].set_xlim(0,30)
ax[0].set_xlabel(r'$d\ (\mu m)$')
ax[0].set_ylabel(r'$P$')
ax[0].legend()


for i in range(num_batch):
    expectation = np.sum( (bins[1:]-0.5)*aprf[i,:] )
    label=case[i]+': runs'+str(batchs[i])+'_mean'+str(round(expectation, 3))
    ax[1].plot(aprf[i,:], line_styles[i], label=label)
ax[1].set_xlim(0,40)    
ax[1].set_xlabel(r'$Asp$')
ax[1].set_ylabel(r'$P$')
ax[1].legend()

fig.savefig('rom_qois.pdf',dpi=600)


