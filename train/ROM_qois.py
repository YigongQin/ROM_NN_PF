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

def ROM_qois(nx,ny,dx,G,angle_id,tip_y,frac,Ni0,Nif,area0,areaf):

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
        
   # step 3: attach every grain to the size list
    for g in range(G):
         
        if piece_len[g,0]>0:
            list0g = area0[angle_id[g]]
            listfg = areaf[angle_id[g]]
            list0g.extend([ar0[g]])
            listfg.extend([arf[g]])
        #print(area0[angle_id[g]])
        #print(areaf[angle_id[g]])
# plot a bar graph with initial and final grain information:
# (1) the number of grains active on the S-L interface: total/num_simulations
# (2) the average and standard deviation of the grain area: total/num_grains

batch = 900#2500
num_batch = 4
num_runs = batch*num_batch

frames = 27 +1
G = 20
bars = 9
pfs=11
# targets
Ni0 = np.zeros(bars,dtype=int) 
Nif = np.zeros(bars,dtype=int)    
di0 = np.zeros(bars)
dif = np.zeros(bars)   
di0_std = np.zeros(bars)
dif_std = np.zeros(bars)
# create list of lists, both of them should have the same size 
# size of each interval is Ni0, for each sublist, do mean and std
area0 = [[] for _ in range(bars)]
areaf = [[] for _ in range(bars)]


## start with loading data

datasets = glob.glob('../../*test250_Mt70536*.h5')
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


for batch_id in range(num_batch):
  fname =datasets[batch_id]; print(fname)
  f = h5py.File(str(fname), 'r')
  aseq_asse = np.asarray(f['sequence'])
  frac_asse = np.asarray(f['fractions'])
  tip_y_asse = np.asarray(f['y_t'])
  angles_asse = np.asarray(f['angles'])
  #number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
  #print(number_list[6])
  # compile all the datasets interleave
  for run in range(batch):
 # for run in range(1):
    aseq = aseq_asse[run*G:(run+1)*G]  # 1 to 10
    frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
    tip_y = tip_y_asse[run*frames:(run+1)*frames]
    angles = angles_asse[run*pfs:(run+1)*pfs]
    #print(frac[:,0]) 
    Color = angles[aseq]*180/pi+90  # in the range of 0-90 degree
    interval = np.asarray(Color/10,dtype=int)
    #print('angle sequence', interval)
    ROM_qois(nx,ny,dx,G,interval,tip_y,frac,Ni0,Nif,area0,areaf)
    #print(Ni0,Nif)
Ni0=Ni0/num_runs
Nif=Nif/num_runs
print('initial distribution of average number of grains',Ni0)
print('final istribution of average number of grains',Nif)
#print(area0)

for i in range(bars):
#    print('Initial # grains:',len(area0[i]),' average:',dx**2*np.mean(np.asarray(area0[i])))
#    print('Final # grains:',len(areaf[i]),' average:',dx**2*np.mean(np.asarray(areaf[i])))
   di0[i] = np.mean( np.sqrt(4.0*np.asarray(area0[i])/pi) )*dx
   dif[i] = np.mean( np.sqrt(4.0*np.asarray(areaf[i])/pi) )*dx
   di0_std[i] = np.std( np.sqrt(4.0*np.asarray(area0[i])/pi) )*dx/np.sqrt(len(area0[i]))
   dif_std[i] = np.std( np.sqrt(4.0*np.asarray(areaf[i])/pi) )*dx/np.sqrt(len(areaf[i]))
print('initial distribution of average grain diameter',di0)
print('final istribution of average grain diameter',dif)
print(di0_std,dif_std)


labels = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90']

x = np.arange(len(labels))
width = 0.35

fig1, ax1 = plt.subplots()


rects1 = ax1.bar(x - width/2, Ni0, width, label=r'$t=0$')
rects2 = ax1.bar(x + width/2, Nif, width, label=r'$t=T_f$')


ax1.set_ylabel('# grains')
ax1.set_xlabel(r'$\theta_0^{\circ}$')
ax1.set_title('average # grains on the S-L interface')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

#ax1.bar_label(rects1, padding=3)
#ax1.bar_label(rects2, padding=3)

fig1.tight_layout()

plt.show()
plt.savefig('num_grains'+str(num_runs)+'.png')


fig2, ax2 = plt.subplots()

rects3 = ax2.bar(x - width/2, di0, width, label=r'$t=0$', yerr=di0_std)
rects4 = ax2.bar(x + width/2, dif, width, label=r'$t=T_f$', yerr=dif_std)


ax2.set_ylabel(r'grain diameter ($\mu m$)')
ax2.set_xlabel(r'$\theta_0^{\circ}$')
ax2.set_title('average grain diameter')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

#ax1.bar_label(rects1, padding=3)
#ax1.bar_label(rects2, padding=3)

fig2.tight_layout()

plt.show()
plt.savefig('grain_dia'+str(num_runs)+'.png')

sio.savemat('qois_'+str(num_runs)+'.mat',{'di0':di0,'dif':dif,'Ni0':Ni0,'Nif':Nif})



