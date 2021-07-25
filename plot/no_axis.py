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
plt.rcParams.update({'font.size': 10})
plt.style.use("dark_background")
#plt.rcParams['text.usetex']=True
#plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'
mathtext.FontConstantsBase.sub1 = 0.2
num_gpu = 1
npx  = 1
npy = npx

#f = h5py.File('data.h5', 'r')

#filebase = sys.argv[1];
#nx = int(sys.argv[2])
#ratio = float(sys.argv[3])
#ny = int(nx*ratio)
ratio=1
hd=1
filebase = sys.argv[1]
filename = filebase + str(num_gpu-1)+'.h5'
f = h5py.File(filename, 'r')
    #data = sio.loadmat(filename,squeeze_me = True)
   # print(ny)
   # field = data[var];nx = data['nx'];ny=data['nz']
x = f['x_coordinates']
y = f['y_coordinates']
fnx = len(x); fny = len(y); nx = fnx-2*hd; ny = fny-2*hd;
print(nx,ny)
lx = 40
ly = lx*ratio
print('the limits for geometry lx, ly: ',lx,ly)

u = np.zeros((nx*npx+1,ny*npy+1))


idx = np.array([0]) #,10,15,20])
var_list = ['Uc','phi','alpha']
range_l = [0,-1,0]
range_h = [5,1,10]


DNSmode = 'nb'   # nb or hg

#macrodata = sio.loadmat('WD_shallow.mat', squeeze_me=True)
t0 = 1.83e-5 #macrodata['t_macro'][-1]




#colors = [(0.2, 0.4, 1, 1),(1, 0., 0., 1),(0.3, 1, 0.3, 1),(0.3, 0.3, 0.3, 1)]

fid=2
var = var_list[fid]
vmin = np.float64(range_l[fid])
vmax = np.float64(range_h[fid])
print('the field variable: ',var,', range (limits):', vmin, vmax)
fg_color='white'; bg_color='black'
for j in range(len(idx)):
  for i in range(num_gpu):
    
    px = i - npx*int(i/npx)
    py = int(i/npx)
    filename = filebase + str(i)+'.h5'
    f = h5py.File(filename, 'r')
    #data = sio.loadmat(filename,squeeze_me = True)
   # print(ny)
   # field = data[var];nx = data['nx'];ny=data['nz']
    x = f['x_coordinates']
    y = f['y_coordinates']
    field = f[var]; fnx = len(x); fny = len(y); nx = fnx-2*hd; ny = fny-2*hd;
    field = np.reshape(field,(fnx,fny),order='F');print('shape of the rank',i,':',field.shape)
    print(i, field[hd:fnx-hd,hd:fny-hd])
    if fid==2: u[px*nx:(px+1)*nx,py*ny:(py+1)*ny] = field[hd:fnx-hd,hd:fny-hd]#*180/pi
    else: u[px*nx:(px+1)*nx,py*ny:(py+1)*ny] = field[hd:fnx-hd,hd:fny-hd]    
    
  fig, ax = plt.subplots()
  time = t0 #idx[j]/20*t0
  if time == 0.0: plt.title('t = 0' + ' s',color=bg_color)
  else: plt.title('t = '+str('%4.2e'%time) + ' s',color=bg_color)
  #axins = inset_axes(ax,width="3%",height="50%",loc='lower left')#,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax.transAxes,borderpad=0,)
  #cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),norm=colors.PowerNorm(gamma=0.3,vmin=vmin, vmax=vmax),origin='lower',extent= (-lx, 0, -ly, 0))
  if fid==2: cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),origin='lower',extent= (0, lx, 0, ly))
  else: cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),origin='lower',extent= ( 0, lx, 0, ly))
  #cs = ax.pcolor(u.T,norm=colors.LogNorm(vmin=vmin, vmax=vmax),cmap=plt.get_cmap('gray'),origin='lower',extent= (-lx, 0, -ly, 0))
 # ax.set_xticks([-lx, -lx/2,0]); ax.set_yticks([-ly, -ly/2,0])
  ax.set_xlabel('$x\ (\mu m)$'); ax.set_ylabel('$y\ (\mu m)$'); 
  ax.spines['bottom'].set_color(bg_color);ax.spines['left'].set_color(bg_color)
  ax.yaxis.label.set_color(bg_color); ax.xaxis.label.set_color(bg_color)
  ax.tick_params(axis='x', colors=bg_color); ax.tick_params(axis='y', colors=bg_color);
  cs.set_clim(vmin, vmax)
  plt.show()
  plt.savefig(filebase+'_'+ var + str(idx[j])+ '.pdf',dpi=800,facecolor="white", bbox_inches='tight')
  plt.close()
  print(u.shape)
  print(u.T)













