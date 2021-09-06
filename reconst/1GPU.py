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
plt.rcParams.update({'font.size': 10})
mathtext.FontConstantsBase.sub1 = 0.

num_gpu = 1
npx  = 1
npy = npx
ratio=1
hd=1
filebase = sys.argv[1]
filename = filebase + str(num_gpu-1)+'.h5'


number_list=re.findall(r"[-+]?\d*\.\d+|\d+", filename)


pfs = int(number_list[0])+1; print('PFs',pfs)
train = int(number_list[1]); print('train',train)
test = int(number_list[2]); print('test',test)
G = int(number_list[4]); print('grains',G)
frames = int(number_list[5])+1; print('grains',frames)

f = h5py.File(filename, 'r')
x = f['x_coordinates']
y = f['y_coordinates']
fnx = len(x); fny = len(y);
length = fnx*fny
lx = 60
ly = lx*ratio
print('the limits for geometry lx, ly: ',lx,ly)

idx = np.array([0]) #,10,15,20])
var_list = ['Uc','phi','alpha']
range_l = [0,-1,0]
range_h = [5,1,90]

fid=2
var = var_list[fid]
vmin = np.float64(range_l[fid])
vmax = np.float64(range_h[fid])
print('the field variable: ',var,', range (limits):', vmin, vmax)
fg_color='white'; bg_color='black'





for tid in range(test):

  alpha_id = (f[var])[tid*length:(tid+1)*length]

#  aseq = np.asarray(f['sequence'])[tid*G:(tid+1)*G]
  aid = tid + train
  angles = np.asarray(f['angles'])[aid*pfs:(aid+1)*pfs]
  print('tid, angle',tid,angles)

  alpha_id = np.asarray(alpha_id).reshape((fnx,fny),order='F')
  u = angles[alpha_id]/pi*180+90


  fig, ax = plt.subplots()
  #time = t0 #idx[j]/20*t0
  #if time == 0.0: plt.title('t = 0' + ' s',color=bg_color)
  #else: plt.title('t = '+str('%4.2e'%time) + ' s',color=bg_color)
  axins = inset_axes(ax,width="3%",height="50%",loc='lower left')#,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax.transAxes,borderpad=0,)
  #cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),norm=colors.PowerNorm(gamma=0.3,vmin=vmin, vmax=vmax),origin='lower',extent= (-lx, 0, -ly, 0))
  if fid==2: cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),origin='lower',extent= (0, lx, 0, ly))
  else: cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),origin='lower',extent= ( 0, lx, 0, ly))
  #cs = ax.pcolor(u.T,norm=colors.LogNorm(vmin=vmin, vmax=vmax),cmap=plt.get_cmap('gray'),origin='lower',extent= (-lx, 0, -ly, 0))
 # ax.set_xticks([-lx, -lx/2,0]); ax.set_yticks([-ly, -ly/2,0])
  ax.set_xlabel('$x\ (\mu m)$'); ax.set_ylabel('$y\ (\mu m)$'); 
  ax.spines['bottom'].set_color(bg_color);ax.spines['left'].set_color(bg_color)
  ax.yaxis.label.set_color(bg_color); ax.xaxis.label.set_color(bg_color)
  ax.tick_params(axis='x', colors=bg_color); ax.tick_params(axis='y', colors=bg_color);
  #plt.locator_params(nbins=3)
  cbar = fig.colorbar(cs,cax = axins)#,ticks=[1, 2, 3,4,5])
  cbar.set_label(r'$\alpha$', color=fg_color)
  cbar.ax.yaxis.set_tick_params(color=fg_color)
  cbar.outline.set_edgecolor(fg_color)
  plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=fg_color)
  cs.set_clim(vmin, vmax)
  plt.show()
  plt.savefig(var + 'test_' +str(idx[tid])+ '.pdf',dpi=800,facecolor="white", bbox_inches='tight')
  plt.close()
  print(u.shape)
  print(u.T)













