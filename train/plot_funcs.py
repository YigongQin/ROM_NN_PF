#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:52:09 2021

@author: yigongqin
"""
import numpy as np
import scipy.io as sio
import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
var_list = ['Uc','phi','alpha']
range_l = [0,-5,0]
range_h = [5,5,90]
fid=2
var = var_list[fid]
vmin = np.float64(range_l[fid])
vmax = np.float64(range_h[fid])
plt.rcParams.update({'font.size': 10})
#plt.style.use("dark_background")
mathtext.FontConstantsBase.sub1 = 0.2  
fg_color='white'; bg_color='black'

def subplot_rountine(fig, ax, cs, idx):
    
      ax.set_xlabel('$x\ (\mu m)$'); 
      if idx ==1: ax.set_ylabel('$y\ (\mu m)$');
      ax.spines['bottom'].set_color(bg_color);ax.spines['left'].set_color(bg_color)
      ax.yaxis.label.set_color(bg_color); ax.xaxis.label.set_color(bg_color)
      ax.tick_params(axis='x', colors=bg_color); ax.tick_params(axis='y', colors=bg_color);
      if idx==1:
        axins = inset_axes(ax,width="3%",height="50%",loc='lower left')
        cbar = fig.colorbar(cs,cax = axins)#,ticks=[1, 2, 3,4,5])
        cbar.set_label(r'$\alpha_0$', color=fg_color)
        cbar.ax.yaxis.set_tick_params(color=fg_color)
        cbar.outline.set_edgecolor(fg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=fg_color)
      cs.set_clim(vmin, vmax)
    
      return

def plot_IO(anis,G0,Rmax,G,x,y,aseq,tip_y,alpha_true,frac, plot_idx,ymax,final,pf_angles, area_true, area):

    #print('angle sequence', aseq)
    #print(frac) 
    xmin = x[1]; xmax = x[-2]
    ymin = y[1]; ytop = y[-2]
    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    dx = x[1]-x[0]
    nt=len(tip_y)
    #input_frac = int((window-1)/(nt-1)*100)
    alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')    

    ntip_y = np.asarray(tip_y/dx,dtype=int)
    
    p_len = np.asarray(np.round(frac*nx),dtype=int)
    piece_len = np.cumsum(p_len,axis=0)
    correction = piece_len[-1, :] - fnx
    for g in range(G//2, G):
      piece_len[g,:] -= correction

    piece0 = piece_len[:,0]
    #print(piece_len[-1,:])
    field = np.zeros((nx,ny),dtype=int)
    ini_field = np.zeros((nx,ny),dtype=int)


            

#=========================start fill the initial field=================

    temp_piece = np.zeros(G, dtype=int)
    for j in range(ntip_y[0]):
     #  start with temp_piece
       for g in range(G):
          if j <= ntip_y[0]: temp_piece[g] = piece0[g]
          else:
            fint = interp1d(ntip_y, piece_len[g,:],kind='linear')
            new_f = fint(j)
            temp_piece[g] = np.asarray(new_f,dtype=int)

       for g in range(G):
        if g==0:
          for i in range( temp_piece[g]):
            if (i>nx-1 or j>ny-1): break
            ini_field[i,j] = aseq[g]

        else:
          for i in range(temp_piece[g-1], temp_piece[g]):
            if (i>nx-1 or j>ny-1): break         
            ini_field[i,j] = aseq[g]


#=========================start fill the final field=================
    nymax = int(ymax/dx)
    temp_piece = np.zeros(G, dtype=int)
    miss=0
    for j in range(ntip_y[final-1]):
     #  loc = 0
       for g in range(G):
          if j <= ntip_y[0]: temp_piece[g] = piece0[g]
          else:
            fint = interp1d(ntip_y, piece_len[g,:],kind='linear')
            new_f = fint(j)
            temp_piece[g] = np.asarray(new_f,dtype=int)
       #print(temp_piece)
       #temp_piece = np.asarray(np.round(temp_piece/np.sum(temp_piece)*nx),dtype=int)
       for g in range(G):
        if g==0:
          for i in range( temp_piece[g]):
            if (i>nx-1 or j>ny-1): break
           # print(loc)
            field[i,j] = aseq[g]
            if (alpha_true[i+1,j+1]!=field[i,j]) and j< nymax: miss+=1
        else:
          for i in range(temp_piece[g-1], temp_piece[g]):
            if (i>nx-1 or j>ny-1): break
           # print(loc)
            field[i,j] = aseq[g]
            if (alpha_true[i+1,j+1]!=field[i,j]) and j< nymax: miss+=1


#=========================start fill the extra field=================
    for g in range(G):

      if p_len[g, -1] ==0: height =0 
      else: height = int(area[g]/p_len[g, -1])
      #print(height)
      for j in range(ntip_y[-1], ntip_y[-1]+height):

        if g==0:
          for i in range( temp_piece[g]):
            if (i>nx-1 or j>ny-1): break
            field[i,j] = aseq[g]

        else:
          for i in range(temp_piece[g-1], temp_piece[g]):
            if (i>nx-1 or j>ny-1): break         
            field[i,j] = aseq[g]


#========================start plotting area, plot ini_field, alpha_true, and field======

    ## count for the error of y
    
    if nymax-ntip_y[final-1]>0: miss += nx*(nymax-ntip_y[final-1])

    ## count for the error of area
    for g in range(G):
      miss += np.absolute(area[g]-area_true[g])
      #print(area[g], area_true[g])
    miss_rate = miss/( nx*nymax + np.sum(area_true) );

   # error=field-alpha_true[1:-1,1:-1]
    plot_flag=True
    if plot_flag==True:
      fig = plt.figure()
      txt = r'$\epsilon_k$'+str(anis)+'_G'+str("%1.1f"%G0)+r'_$R_{max}$'+str(Rmax)
      fig.text(.5, .2, txt, ha='center')
      ax1 = fig.add_subplot(131)
      cs1 = ax1.imshow(pf_angles[ini_field].T,cmap=plt.get_cmap('jet'),origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax1, cs1, 1)
      #ax1.set_title('input:'+str(input_frac)+'%history',color=bg_color,fontsize=8)
      ax1.set_title('initial condition',color=bg_color,fontsize=8)
      ax2 = fig.add_subplot(132)
      cs2 = ax2.imshow(pf_angles[alpha_true[1:-1,1:-1]].T,cmap=plt.get_cmap('jet'),origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax2, cs2, 2)
      ax2.set_title('final:PDE_solver', color=bg_color,fontsize=8)
      
      ax3 = fig.add_subplot(133)
      cs3 = ax3.imshow(pf_angles[field].T,cmap=plt.get_cmap('jet'),origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax3, cs3, 3)
      ax3.set_title('final:NN_predict_'+str(int(miss_rate*100))+'%error', color=bg_color, fontsize=8)
      
      plt.savefig(var + '_grains' + str(G) + '_case' + str(plot_idx)+ '_error'+ str("%d"%int(miss_rate*100)) +'.png',dpi=800,facecolor="white", bbox_inches='tight')
      plt.close()

    
    return


def miss_rate(anis,G0,Rmax,G,x,y,aseq,tip_y,alpha_true,frac, plot_idx,ymax,final,pf_angles, area_true, area):
    

    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    dx = x[1]-x[0]
    alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')    
    ntip_y = np.asarray(tip_y/dx,dtype=int)   
    piece_len = np.asarray(np.round(frac*nx),dtype=int)
    piece_len = np.cumsum(piece_len,axis=0)
    correction = piece_len[-1, :] - fnx
    for g in range(G//2, G):
      piece_len[g,:] -= correction

    piece0 = piece_len[:,0]
    field = np.zeros((nx,ny),dtype=int)


#=========================start fill the final field=================
    nymax = int(ymax/dx)
    temp_piece = np.zeros(G, dtype=int)
    miss=0
    for j in range(ntip_y[final-1]):
     #  loc = 0
       for g in range(G):
          if j <= ntip_y[0]: temp_piece[g] = piece0[g]
          else:
            fint = interp1d(ntip_y, piece_len[g,:],kind='linear')
            new_f = fint(j)
            temp_piece[g] = np.asarray(new_f,dtype=int)
       #print(temp_piece)
       #temp_piece = np.asarray(np.round(temp_piece/np.sum(temp_piece)*nx),dtype=int)
       for g in range(G):
        if g==0:
          for i in range( temp_piece[g]):
            if (i>nx-1 or j>ny-1): break
            field[i,j] = aseq[g]
            if (pf_angles[alpha_true[i+1,j+1]]!=pf_angles[field[i,j]]) and j< nymax: miss+=1
        else:
          for i in range(temp_piece[g-1], temp_piece[g]):
            if (i>nx-1 or j>ny-1): break
           # print(loc)
            field[i,j] = aseq[g]
            if (pf_angles[alpha_true[i+1,j+1]]!=pf_angles[field[i,j]]) and j< nymax: miss+=1
            
    ## count for the error of y
    miss_frac = miss

    if nymax-ntip_y[final-1]>0: miss += nx*(nymax-ntip_y[final-1])

    ## count for the error of area
    miss_area = 0
    for g in range(G):
      miss += np.absolute(area[g]-area_true[g])
      miss_area += np.absolute(area[g]-area_true[g])

    all_area = nx*nymax + np.sum(area_true) 
    miss_rate = miss/all_area
 
    print('component: frac', miss_frac/miss, ', y', nx*(nymax-ntip_y[final-1])/miss, ', area', miss_area/miss)
    return miss_rate

