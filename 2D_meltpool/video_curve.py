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
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0, 1, 256))
pink = np.array([255/256, 255/256, 210/256, 1])
newcolors[0, :] = pink
newcmp = ListedColormap(newcolors)
import matplotlib.transforms as mtransforms
fg_color='white'; bg_color='black'



ft=24
plt.rcParams.update({'font.size': ft})
mathtext.FontConstantsBase.sub1 = 0.


tid = 0 #int(sys.argv[1])
Rmax = 1.52

batch = 1
num_gpu = 1
npx  = 1
npy = npx
ratio=1

#datasets = sorted(glob.glob('../../*frames75*6933304*76500*.h5'))
datasets = sorted(glob.glob('*6933304*76500*.h5'))
filename = datasets[0]
number_list=re.findall(r"[-+]?\d*\.\d+|\d+", filename)


pfs = int(number_list[0])+1; print('PFs',pfs)
train = int(number_list[1]); print('train',train)
test = int(number_list[2]); print('test',test)
G = int(number_list[3]); print('grains',G)
frames = int(number_list[4])+1; print('frames',frames)

f = h5py.File(filename, 'r')
x = np.asarray(f['x_coordinates'])
y = np.asarray(f['y_coordinates'])

dx = x[2]-x[1]
fnx = len(x); 
fny = len(y);
#fny = fnx - 1
length = fnx*fny
lx = 60
ly = lx*ratio
print('the limits for geometry lx, ly: ',lx,ly)



r_in = 80/pi*2
r_out = (80+2.2528)/pi*2

rn = int( r_in*(fnx-2)/60 )
rn_out = int( r_out*(fnx-2)/60 )

phi = sio.loadmat('phi_square.mat')['phi']
#phi = sio.loadmat('phi_square'+f'{tid:03}'+'.mat')['phi']

var_list = ['Uc','phi','alpha']
range_l = [0,-1,0]
range_h = [5,1,90]

fid=2
var = var_list[fid]
vmin = np.float64(range_l[fid])
vmax = np.float64(range_h[fid])
print('the field variable: ',var,', range (limits):', vmin, vmax)



alpha_id = (f[var])[tid*length:(tid+1)*length]
aid = tid + train
angles = np.asarray(f['angles'])[:pfs]

  
alpha = alpha_id.reshape((fnx,fny),order='F')[1:-1,1:-1]
alpha = ( angles[alpha]/pi*180 + 90 )*(alpha>0)
  

nx = fnx-2
ny = fny-2

mask_ratio = 0.1 
mask_int = int(mask_ratio*nx)

xx, yy = np.mgrid[:nx, :ny]
circle = (xx-nx)**2 + (yy-ny)**2


nxp = phi.shape[0]
nyp = phi.shape[1]
xp, yp = np.mgrid[:nxp, :nyp]
theta = xp/nxp*pi/2
r = int(round(r_out/dx ))- yp
x_data = (nx - r*np.cos(theta)).flatten()
y_data = (ny - r*np.sin(theta)).flatten()
points = np.zeros((x_data.shape[0],2))
points[:,0] = x_data
points[:,1] = y_data
values = ( angles[phi]/pi*180 + 90 )*(phi>0)
values = values.flatten()
interp_a = griddata(points, values, (xx, yy), method='nearest')


fig, ax = plt.subplots(1,3,figsize=(20,8))
case = ['PDE','GrainNN','MR 9%']
for i in range(3):
    alpha = np.asarray(alpha_id).reshape((fnx,fny),order='F')[1:-1,1:-1]
    alpha = ( angles[alpha]/pi*180 + 90 )*(alpha>0)
    u = alpha
    if i==1:
        u=interp_a
    if i==2: 
        u = 1.0*(alpha!=interp_a)
        newcmp = 'Reds'


    if i>=0:
        u[circle>rn**2] = np.NaN 
    else:
        u[rn_out**2<circle] = np.NaN   
        u[rn**2>circle] = 0
 
    if i==2:
        no_nonnan = nx*ny - np.sum(1*(np.isnan(u)))
        no_one = np.sum(1*(u==1.0))
        MR = no_one/no_nonnan
        print('MR', MR)
    cs = ax[i].imshow(u.T[mask_int:,mask_int:],cmap=newcmp,origin='lower', \
        extent= ( lx*mask_ratio, lx,  ly*mask_ratio, ly))
    #trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
 
    if i!=0: ax[i].axis('off')
    else: 
        ax[i].yaxis.set_ticks([20,40,60]); ax[i].xaxis.set_ticks([20,40,60]); 
        ax[i].set_xlabel(r'$x (\mu m)$');ax[i].set_ylabel(r'$y (\mu m)$');
    if i==0:   
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax[i].text(0.45,0.05, r'$t=$'+str("%1.2f"%(60/Rmax*tid/100))+r'$\mu s$', transform=ax[i].transAxes + trans, fontsize=ft, horizontalalignment='center')

    if i<2: ax[i].set_title(case[i],fontsize=28)
    else: ax[i].set_title('Error '+str(int(MR*100))+'%',color=bg_color,fontsize=28)
        
    if i==0:
        axins = inset_axes(ax[i],width="3%",height="30%",loc='lower left')#,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax[i].transax[i]es,borderpad=0,)
        cbar = fig.colorbar(cs,cax = axins)#,ticks=[1, 2, 3,4,5])
        cbar.set_label(r'$\alpha_0$', color=bg_color)
        cbar.ax.yaxis.set_tick_params(color=bg_color)
        cbar.outline.set_edgecolor(bg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=bg_color)
        cs.set_clim(vmin, vmax)
    if i==3:
        cs.set_clim(0,1)
    print(u.shape)
    print(u.T)



    plt.savefig(var + '_grains' + str(G) + 'meltpool_frame' + f'{tid:03}'+'.png',dpi=400,facecolor="white", bbox_inches='tight')
    









