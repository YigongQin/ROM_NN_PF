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

def plot_real(x,y,alpha_true):
    xmin = x[1]; xmax = x[-2]
    ymin = y[1]; ymax = y[-2]
    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')
    var_list = ['Uc','phi','alpha']
    range_l = [0,-1,0]
    range_h = [5,1,10]
    fid=2
    var = var_list[fid]
    vmin = np.float64(range_l[fid])
    vmax = np.float64(range_h[fid])
    print('the field variable: ',var,', range (limits):', vmin, vmax)
    fg_color='white'; bg_color='black'
   # error=field-alpha_true[1:-1,1:-1]
    plot_flag=True
    if plot_flag==True:
      fig, ax = plt.subplots()
      axins = inset_axes(ax,width="3%",height="50%",loc='lower left')#,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax.transAxes,borderpad=0,)
      #cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),norm=colors.PowerNorm(gamma=0.3,vmin=vmin, vmax=vmax),origin='lower',extent= (-lx, 0, -ly, 0))
      #cs = ax.imshow((field).T,cmap=plt.get_cmap('jet'),origin='lower',extent= (xmin,xmax, ymin, ymax))
      cs = ax.imshow((alpha_true[1:-1,1:-1]).T,cmap=plt.get_cmap('jet'),origin='lower',extent= (xmin,xmax, ymin, ymax))
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
      #plt.show()
      plt.savefig('true'+'_'+ var + '.png',dpi=800,facecolor="white", bbox_inches='tight')
      plt.close()
 
    


def plot_reconst(G,x,y,aseq,tip_y,alpha_true,frac):
    plt.rcParams.update({'font.size': 10})
    plt.style.use("dark_background")
    #plt.rcParams['text.usetex']=True
    #plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'
    mathtext.FontConstantsBase.sub1 = 0.2    
    xmin = x[1]; xmax = x[-2]
    ymin = y[1]; ymax = y[-2]
    print('xmin',xmin,'xmax',xmax,'ymin',ymin,'ymax',ymax)
    dx = x[1]-x[0]
    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    nt=len(tip_y); len_seq=nt
    alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')
    print('nx,ny,nt', nx,ny,nt)
    print('angle sequence', aseq)
    
    ntip_y = np.asarray(tip_y/dx,dtype=int)
    print('ntip',ntip_y)
    
    piece_len = np.asarray(np.round(frac*nx),dtype=int)
    #piece_len = np.reshape(piece_len,(G,len_seq), order='F')
    piece_len = np.cumsum(piece_len,axis=0)
    piece0 = piece_len[:,0]
    print('len_piece', piece_len)
    
    field = np.zeros((nx,ny),dtype=int)
    
    
    temp_piece = np.zeros(G, dtype=int)
    miss=0
    for j in range(ntip_y[-1]):
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
           # print(loc)
            field[i,j] = aseq[g]
            if (alpha_true[i+1,j+1]!=field[i,j]): miss+=1
        elif g==G-1:
          for i in range(temp_piece[g-1], nx):
            field[i,j] = aseq[g]
            if (alpha_true[i+1,j+1]!=field[i,j]): miss+=1
        else:
          for i in range(temp_piece[g-1], temp_piece[g]):
           # print(loc)
            field[i,j] = aseq[g]
            if (alpha_true[i+1,j+1]!=field[i,j]): miss+=1
         #   loc+=1
          #  if (i>nx-1): break
    
    print(field)
    print('miss rate', miss/(nx*ntip_y[-1]))
    
    var_list = ['Uc','phi','alpha']
    range_l = [0,-5,0]
    range_h = [5,5,10]
    fid=2
    var = var_list[fid]
    vmin = np.float64(range_l[fid])
    vmax = np.float64(range_h[fid])
    print('the field variable: ',var,', range (limits):', vmin, vmax)
    fg_color='white'; bg_color='black'
   # error=field-alpha_true[1:-1,1:-1]
    plot_flag=True
    if plot_flag==True:
      fig, ax = plt.subplots()
      axins = inset_axes(ax,width="3%",height="50%",loc='lower left')#,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax.transAxes,borderpad=0,)
      #cs = ax.imshow(u.T,cmap=plt.get_cmap('jet'),norm=colors.PowerNorm(gamma=0.3,vmin=vmin, vmax=vmax),origin='lower',extent= (-lx, 0, -ly, 0))
      cs = ax.imshow((field).T,cmap=plt.get_cmap('jet'),origin='lower',extent= (xmin,xmax, ymin, ymax))
      #cs = ax.imshow((field-alpha_true[1:-1,1:-1]).T,cmap=plt.get_cmap('jet'),origin='lower',extent= (xmin,xmax, ymin, ymax))
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
      #plt.show()
      plt.savefig('recons'+'_'+ var + '.png',dpi=800,facecolor="white", bbox_inches='tight')
      plt.close()

    
    
    return
