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
noise = eps

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
            width = np.max(piece_len[g,:])
            apf[g] = height/( arf[g]/height )
          #  apf[g] = height/width
            
        
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

            if np.all(piece_len[g,:]>noise):
                height = ntip_y[-1] + extra_area[g,-1]/piece_len[g,-1]
            else: 
                height_id = list((piece_len[g,:]>noise)*1).index(0)-1
                height = ntip_y[height_id]
              
            width = np.max(piece_len[g,:])
            apf[g] = height/( arf[g]/height )
           # apf[g] = height/width
        
   # step 4: attach every grain to the size list
    for g in range(G):
         
        if piece_len[g,0]>eps:

            area0.extend(list([ar0[g]]))
            areaf.extend(list([arf[g]]))
            ap_list.extend(list( [apf[g]]))


batch = 1000#2500
num_batch = 8
num_runs = batch*num_batch

frames = 24 +1
G = 8
bars = 100
# targets

mode = 'self'
mode = 'synthetic'

datasets = sorted(glob.glob('*PF8*154272*.h5'))
synthsets = sorted(glob.glob('synthetic*.mat'))

print(datasets)
print(synthsets)
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

dif = np.zeros((num_batch,bars))
dit = np.zeros((num_batch,bars))     

 
aprf = np.zeros((num_batch,bars)) 
aprt = np.zeros((num_batch,bars)) 

y_t = np.zeros((num_batch, frames))
df_t = np.zeros((num_batch, frames-1))
area_t = np.zeros((num_batch, frames))

y_t_check = np.zeros((num_batch, frames))
df_t_check = np.zeros((num_batch, frames-1))
area_t_check = np.zeros((num_batch, frames))

G0 = np.zeros(num_batch)
anis = np.zeros(num_batch)
Rmax = np.zeros(num_batch)

batchs = np.zeros(num_batch,dtype=int)

bins = np.arange(bars+1)

ks_d = np.zeros(num_batch)
ks_a = np.zeros(num_batch)
err_d = np.zeros(num_batch)
err_a = np.zeros(num_batch)

exp_d = np.zeros(num_batch)
expectation_d = np.zeros(num_batch)
exp_a = np.zeros(num_batch)
expectation_a = np.zeros(num_batch)


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


    synth_dat = sio.loadmat(synthsets[batch_id],squeeze_me=True)
    frac_s = synth_dat['frac']
    assert frac_s.shape[0] == batch
    y_s = synth_dat['y']
    area_s = synth_dat['area']
    
    print('y_average, true', y_t[batch_id,-1], np.mean(y_s, axis=0)[-1] )
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
        extra_area = (extra_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
    
        if mode == 'self':
            frac_rom = frac
            tip_y_rom = tip_y
            extra_area_rom = extra_area

        elif mode == 'synthetic':

            frac_rom = frac_s[run,:,:].T
            tip_y_rom = y_s[run,:]
            extra_area_rom = area_s[run,:,:].T
        else: raise ValueError('mode not right')
        interval = np.asarray(aseq/10,dtype=int)
        assert np.all(interval>=0) and np.all(interval<9)
        #print('angle sequence', interval)

        total_area = (total_area_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')
        tip_y_f    = (tip_y_f_asse   [run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')        
        ROM_qois(nx,ny,dx,G,interval,tip_y_rom,frac_rom,extra_area_rom,                    Ni0,Nif,area0,areaf,apr_list)
        true_qois(nx,ny,dx,G,interval,tip_y,frac,extra_area,total_area,tip_y_f,Ni0,Nif,area0_t,areaf_t,apr_list_t)
        
        df_t_check[batch_id,:] += np.mean(np.absolute(np.diff(frac_rom, axis=1 )), axis = 0)
        area_t_check[batch_id,:] += np.mean(extra_area_rom, axis = 0)
        y_t_check[batch_id,:] += tip_y_rom
        
    df_t_check[batch_id,:] /= batch
    y_t_check[batch_id,:] /= batch
    area_t_check[batch_id,:] /= batch
    
    
    di_f = np.sqrt(4.0*np.asarray(areaf)/pi)*dx
    apr_list = np.asarray(apr_list)
    dif[batch_id,:], _ = np.histogram(di_f , bins, density=True)
    aprf[batch_id,:], _ = np.histogram(apr_list, bins, density=True)

    di_t = np.sqrt(4.0*np.asarray(areaf_t)/pi)*dx 
    apr_list_t = np.asarray(apr_list_t)
    dit[batch_id,:], _ = np.histogram(di_t , bins, density=True)
    aprt[batch_id,:], _ = np.histogram(apr_list_t, bins, density=True)

    batchs[batch_id] = batch


    ks_d[batch_id]= stats.ks_2samp(di_f, di_t)[0]
    ks_a[batch_id]= stats.ks_2samp(apr_list, apr_list_t)[0]

    expectation_d[batch_id] = np.mean(di_f)
    expectation_a[batch_id] = np.mean(apr_list)
    exp_d[batch_id] = np.mean(di_t)
    exp_a[batch_id] = np.mean(apr_list_t)
    err_d[batch_id] = np.absolute(exp_d[batch_id] - expectation_d[batch_id] )/exp_d[batch_id]
    err_a[batch_id] = np.absolute(exp_a[batch_id] - expectation_a[batch_id] )/exp_a[batch_id]

print('parameters')
print(G0)
print(anis)   
print(Rmax) 

print('errors')
np.set_printoptions(precision=3)
print('KSd',ks_d)
print('KSa',ks_a)
print('Errd',err_d)
print('Erra',err_a)


#######################

   ## plotting area

######################



line_styles = ['b-','r--','c-','g--','bs','rs-','cs','gs-']
t = np.linspace(0,90,25)


rom_plot = False

if rom_plot ==True: 
    y_t = y_t_check
    df_t = df_t_check
    area_t = area_t_check

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

if rom_plot == True:  fig.savefig('y_w_rom.pdf',dpi=600)
else: fig.savefig('y_w.pdf',dpi=600)

if rom_plot == False: 
    dif = dit
    aprf = aprt
    expectation_a = exp_a
    expectation_d = exp_d

fig,ax = plt.subplots(1,2,figsize=(15,5))

for i in range(num_batch):
    label=case[i]+': runs'+str(batchs[i])+'_mean'+str(round(expectation_d[i], 3))
    ax[0].plot(dif[i,:], line_styles[i], label=label)
ax[0].set_xlim(0,30)
ax[0].set_xlabel(r'$d\ (\mu m)$')
ax[0].set_ylabel(r'$P$')
ax[0].legend()


for i in range(num_batch):
    label=case[i]+': runs'+str(batchs[i])+'_mean'+str(round(expectation_a[i], 3))
    ax[1].plot(aprf[i,:], line_styles[i], label=label)
ax[1].set_xlim(0,30)    
ax[1].set_xlabel(r'$Asp$')
ax[1].set_ylabel(r'$P$')
ax[1].legend()


if rom_plot == True:  fig.savefig('qoi_rom.pdf',dpi=600)
else: fig.savefig('qoi.pdf',dpi=600)



'''
#  CDF plot

fig,ax = plt.subplots(1,2,figsize=(15,5))

for i in range(4,5):
    ax[0].plot(np.cumsum(dif[i,:]), line_styles[i], label='rom')
    ax[0].plot(np.cumsum(dit[i,:]), line_styles[i+1], label='true')
ax[0].set_xlim(0,30)
ax[0].set_xlabel(r'$d\ (\mu m)$')
ax[0].set_ylabel(r'$P$')
ax[0].legend()


for i in range(4,5):

    ax[1].plot(np.cumsum(aprf[i,:]), line_styles[i], label='rom')
    ax[1].plot(np.cumsum(aprt[i,:]), line_styles[i+1], label='true')

ax[1].set_xlim(0,40)    
ax[1].set_xlabel(r'$Asp$')
ax[1].set_ylabel(r'$P$')
ax[1].legend()

'''


