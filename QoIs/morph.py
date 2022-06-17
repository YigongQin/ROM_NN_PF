#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:08:00 2020

@author: yigongqin
"""
from math import pi
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import matplotlib.mathtext as mathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
import glob, os, re
plt.rcParams.update({'font.size': 32})
mathtext.FontConstantsBase.sub1 = 0.


batch = 1
num_gpu = 1
npx  = 1
npy = npx
ratio=4
hd=1
datasets_old = sorted(glob.glob('../../ML_PF8*.h5'))
rearrange = np.array([0])
datasets=[]
for dat in rearrange:
    datasets.append(datasets_old[dat])
filename = datasets[0]
f0 = h5py.File(str(datasets[0]), 'r')

number_list=re.findall(r"[-+]?\d*\.\d+|\d+", filename)


pfs = int(number_list[0])+1; print('PFs',pfs)
train = int(number_list[1]); print('train',train)
test = int(number_list[2]); print('test',test)
G = int(number_list[3]); print('grains',G)
frames = int(number_list[4])+1; print('frames',frames)

f = h5py.File(filename, 'r')
x = f['x_coordinates']
y = f['y_coordinates']
fnx = len(x); 
fny = len(y);
#fny = fnx - 1
length = fnx*fny
lx = 20
ly = lx*ratio
print('the limits for geometry lx, ly: ',lx,ly)

var_list = ['Uc','phi','alpha']
range_l = [0,-1,0]
range_h = [5,1,90]

fid=2
var = var_list[fid]
vmin = np.float64(range_l[fid])
vmax = np.float64(range_h[fid])
print('the field variable: ',var,', range (limits):', vmin, vmax)
fg_color='white'; bg_color='black'


tid_arr = [0,4,8,12,16,20]
case = ['a','b','c','d','e','f','g','h']
fig, ax = plt.subplots(1,8,figsize=(20,12))

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0, 1, 256))
pink = np.array([255/256, 255/256, 210/256, 1])
newcolors[0, :] = pink
newcmp = ListedColormap(newcolors)
import matplotlib.transforms as mtransforms
for i in range(len(tid_arr)):

    fname =datasets[i]; print(fname)
    f = h5py.File(str(fname), 'r')
    number_list=re.findall(r"[-+]?\d*\.\d+|\d+", fname)
    R= float(number_list[7])
    G = float(number_list[6])
    anis = float(number_list[5])
    
    tid= tid_arr[i]
    alpha_id = (f[var])[tid*length:(tid+1)*length]
    aid = 0
    angles = np.asarray(f['angles'])[aid*pfs:(aid+1)*pfs]
  
    u = np.asarray(alpha_id).reshape((fnx,fny),order='F')[1:-1,1:-1]
    u = ( angles[u]/pi*180 + 90 )*(u>0)
  
    
    #time = t0 #idx[j]/20*t0
    #if time == 0.0: plt.title('t = 0' + ' s',color=bg_color)
    #else: plt.title('t = '+str('%4.2e'%time) + ' s',color=bg_color)
    
    cs = ax[i].imshow(u.T,cmap=newcmp,origin='lower',extent= ( 0, lx, 0, ly))
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax[i].text(0.5,1,case[i], transform=ax[i].transAxes + trans, fontsize='large', horizontalalignment='center', verticalalignment='top')
    #ax[i].set_xlabel('('+case[i]+')'+' $G$'+str(int(G))+'_$R$'+str('%1.1f'%R)+'_$\epsilon_k$'+str('%1.2f'%anis)); 
    #ax[i].set_ylabel('$y\ (\mu m)$'); 
    ax[i].spines['bottom'].set_color(bg_color);ax[i].spines['left'].set_color(bg_color)
    ax[i].yaxis.label.set_color(bg_color); ax[i].xaxis.label.set_color(bg_color)
    ax[i].tick_params(axis='x', colors=bg_color); ax[i].tick_params(axis='y', colors=bg_color);
    if i!=0: ax[i].yaxis.set_ticks([]);ax[i].xaxis.set_ticks([]);
    else: ax[i].yaxis.set_ticks([0,30,60]);ax[i].set_xlabel(r'$x (\mu m)$');ax[i].set_ylabel(r'$y (\mu m)$');
    #plt.locator_params(nbins=3)
    if i==0:
        axins = inset_axes(ax[i],width="3%",height="50%",loc='upper left')#,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax[i].transax[i]es,borderpad=0,)
        cbar = fig.colorbar(cs,cax = axins)#,ticks=[1, 2, 3,4,5])
        cbar.set_label(r'$\alpha_0$', color=bg_color)
        cbar.ax.yaxis.set_tick_params(color=bg_color)
        cbar.outline.set_edgecolor(bg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=bg_color)
        cs.set_clim(vmin, vmax)
        #plt.show()
    #plt.savefig(filebase + '_8grains_test_' +str(tid)+ '.pdf',dpi=800,facecolor="white", bbox_inches='tight')
    #plt.close()
    print(u.shape)
    print(u.T)



fig.savefig('phase.pdf',dpi=600, bbox_inches='tight')









