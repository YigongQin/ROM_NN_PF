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
import matplotlib.transforms as mtransforms
var_list = ['Uc','phi','alpha']
range_l = [0,-5,0]
range_h = [5,5,90]
fid=2
var = var_list[fid]
vmin = np.float64(range_l[fid])
vmax = np.float64(range_h[fid])
ft=24
plt.rcParams.update({'font.size': ft})
#plt.style.use("dark_background")
mathtext.FontConstantsBase.sub1 = 0.2  
fg_color='white'; bg_color='black'


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0, 1, 256*100))
ly = np.array([255/256, 255/256, 210/256, 1])
newcolors[0, :] = ly
newcmp = ListedColormap(newcolors)

def subplot_rountine(fig, ax, cs, idx):
    

      if idx ==1: 
        ax.yaxis.set_ticks([0,30,60])
       # ax.yaxis.set_ticks([0,30,60,90,120,150])
        ax.set_xlabel(r'$x (\mu m)$')
        ax.set_ylabel(r'$y (\mu m)$')
      else: 
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
      ax.spines['bottom'].set_color(bg_color);ax.spines['left'].set_color(bg_color)
      ax.yaxis.label.set_color(bg_color); ax.xaxis.label.set_color(bg_color)
      ax.tick_params(axis='x', colors=bg_color); ax.tick_params(axis='y', colors=bg_color);
      if idx==1:
        axins = inset_axes(ax,width=0.25,height=2.5,loc='upper left')
        cbar = fig.colorbar(cs,cax = axins)#,ticks=[1, 2, 3,4,5])
        cbar.set_label(r'$\alpha_0$', color=bg_color)
        cbar.ax.yaxis.set_tick_params(color=bg_color)
        cbar.outline.set_edgecolor(bg_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=bg_color)
      if idx==4:
         cs.set_clim(0,1)
      else:
         cs.set_clim(vmin, vmax)
      return

def plot_IO(anis,G0,Rmax,G,x,y,aseq,pf_angles,alpha_true,tip_y,frac,area, final, extra_time, plot_flag, plot_idx):

    #print('angle sequence', aseq)
    #print(frac) 
    xmin = x[1]; xmax = x[-2]
    ymin = y[1]; ytop = y[-2]
    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    dx = x[1]-x[0]
    nt=len(tip_y)
    #input_frac = int((window-1)/(nt-1)*100)
    alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')[1:-1,1:-1]    

    ntip_y = np.asarray(np.round(tip_y/dx),dtype=int)
    
    #p_len = np.asarray(np.round(frac*nx),dtype=int)
    piece_len = np.asarray(np.round(np.cumsum(frac,axis=0)*nx),dtype=int)
    correction = piece_len[-1, :] - fnx
    for g in range(G//2, G):
      piece_len[g,:] -= correction

    piece0 = piece_len[:,0]
    #print(piece_len[-1,:])
    field = np.zeros((nx,ny),dtype=int)
    ini_field = np.zeros((nx,ny),dtype=int)


            

#=========================start fill the initial field=================

    temp_piece = np.zeros(G, dtype=int)
    for j in range(ntip_y[0]+1):

       for g in range(G):
        temp_piece[g] = piece0[g]
        if g==0:
          for i in range( temp_piece[g]):
            if (i>nx-1 or j>ny-1): break
            ini_field[i,j] = aseq[g]

        else:
          for i in range(temp_piece[g-1], temp_piece[g]):
            if (i>nx-1 or j>ny-1): break         
            ini_field[i,j] = aseq[g]


#=========================start fill the final field=================
    field[:,:ntip_y[0]+1] = ini_field[:,:ntip_y[0]+1]
    if plot_idx==0: alpha_true = field
    if plot_idx>0:
      temp_piece = np.zeros((G, ny), dtype=int)
      extra_y = 0
      if extra_time>0: extra_y = int(extra_time*( ntip_y[final] - ntip_y[final-1]))
      y_range = np.arange(ntip_y[0]+1, ntip_y[final-1]+extra_y+1)

      for g in range(G):
        fint = interp1d(ntip_y, piece_len[g,:],kind='linear')
        new_f = fint(y_range)
        temp_piece[g,y_range] = np.asarray(np.round(new_f),dtype=int)


      for j in y_range:
       #  loc = 0
         #print(temp_piece)
         #temp_piece = np.asarray(np.round(temp_piece/np.sum(temp_piece)*nx),dtype=int)
        for g in range(G):
          if g==0:
            for i in range( temp_piece[g, j]):
              if (i>nx-1 or j>ny-1): break
             # print(loc)
              field[i,j] = aseq[g]
             # if (alpha_true[i+1,j+1]!=field[i,j]) and j< nymax: miss+=1
          else:
            for i in range(temp_piece[g-1, j], temp_piece[g, j]):
              if (i>nx-1 or j>ny-1): break
             # print(loc)
              field[i,j] = aseq[g]
             # if (alpha_true[i+1,j+1]!=field[i,j]) and j< nymax: miss+=1

        #  if temp_piece[G-1]<nx-1: miss += nx-1-temp_piece[G-1]   
  #=========================start fill the extra field=================
      y_f = y_range[-1]
      for g in range(G):
        top_layer = temp_piece[g, y_f]- temp_piece[g-1, y_f] if g>0 else temp_piece[0, y_f]#p_len[g, final-1]
        if  top_layer==0: height =0 
        else: height = int(np.round((area[g]/top_layer)))
        #print(height)
        for j in range(y_f, y_f+height):

          if g==0:
            for i in range( temp_piece[g, y_f]):
              if (i>nx-1 or j>ny-1): break
              field[i,j] = aseq[g]

          else:
            for i in range(temp_piece[g-1, y_f], temp_piece[g, y_f]):
              if (i>nx-1 or j>ny-1): break         
              field[i,j] = aseq[g]

    true_count = np.array([np.sum(alpha_true==g) for g in aseq])
    rom_count = np.array([np.sum(field==g) for g in aseq])
    inset = np.array([np.sum( (alpha_true==g)*(field==g) ) for g in aseq])
    dice = 2*inset/(true_count + rom_count)
    print('\n',dice)
#========================start plotting area, plot ini_field, alpha_true, and field======

    ## count for the error of y
    
    #if nymax-ntip_y[final-1]>0: miss += nx*(nymax-ntip_y[final-1])
    #miss += np.absolute(nx*(nymax-ntip_y[final-1]))
    ## count for the error of area
    #for g in range(G):
    #  miss += np.absolute(area[g]-area_true[g])
      #print(area[g], area_true[g])

    y_top = next( i for i,x  in  enumerate(np.mean(alpha_true, axis=0)) if x<1e-5)
    miss_rate = np.sum( alpha_true[:,:y_top]!=field[:,:y_top] )/(nx*y_top)

    if plot_flag==True:
      fig = plt.figure(figsize=(5*xmax/20,12*ytop/80))
      txt = r'$\epsilon_k$'+str(anis)+'_G'+str("%1.1f"%G0)+r'_$R_{max}$'+str(Rmax)
     # fig.text(.5, .2, txt, ha='center')
      #ax1.set_title('input:'+str(input_frac)+'%history',color=bg_color,fontsize=8)
      ax2 = fig.add_subplot(131)
      cs2 = ax2.imshow(pf_angles[alpha_true].T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax2, cs2, 2)
      ax2.set_title('PDE', color=bg_color,fontsize=28)
      trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
      ax2.text(0.45,0.9, r'$t=$'+str("%1.2f"%(60/Rmax*plot_idx/100))+r'$\mu s$', transform=ax2.transAxes + trans, fontsize=ft, horizontalalignment='center')

      ax2.yaxis.set_ticks([0,30,60])
      ax2.xaxis.set_ticks([0,50])
      ax2.set_xlabel(r'$x (\mu m)$')
      ax2.set_ylabel(r'$y (\mu m)$')

      ax3 = fig.add_subplot(132)
      cs3 = ax3.imshow(pf_angles[field].T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax3, cs3, 3)
      ax3.set_title('GrainNN', color=bg_color, fontsize=28)
      
      ax4 = fig.add_subplot(133)
      cs4 = ax4.imshow(1*(alpha_true!=field).T,cmap='Reds',origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax4, cs4, 4)
      ax4.set_title('Error '+str(int(np.round(miss_rate*100)))+'%',color=bg_color,fontsize=28)

      plt.savefig(var + '_grains' + str(G) + '_frame' + f'{plot_idx:03}'+'.png',dpi=400,facecolor="white", bbox_inches='tight')
      plt.close()

    
    return miss_rate, dice


def plot_synthetic(anis,G0,Rmax,G,x,y,aseq,tip_y_a, p_len_a, extra_time, plot_idx,final,pf_angles, area_a, left_grains, nx_small, alpha_true):
    
    #print('angle sequence', aseq)
    #print(frac) 
    xmin = x[1]; xmax = x[-2]
    ymin = y[1]; ytop = y[-2]
    fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
    dx = x[1]-x[0]
    #nt=len(tip_y)
    #input_frac = int((window-1)/(nt-1)*100)

    #print(piece_len[-1,:])
    field = np.zeros((nx,ny),dtype=int)
    angle_field = np.zeros((nx,ny))


    if plot_idx ==0: 
      tip_y = tip_y_a[0,:]
      ntip_y = np.asarray(tip_y/dx,dtype=int)
      alpha_true[:,ntip_y[0]+1:] = 0
      angle_field = alpha_true
#=========================start fill the final field=================
    if plot_idx>0:
      subruns = p_len_a.shape[0]
      for run in range(subruns):

        angles = pf_angles[run,:]
        tip_y = tip_y_a[run,:]
        ntip_y = np.asarray(tip_y/dx,dtype=int)

        p_len = p_len_a[run,:,:].T
        piece_len = np.cumsum(p_len,axis=0)
        correction = piece_len[-1, :] - nx_small
        for g in range(G//2, G):
          piece_len[g,:] -= correction

        piece0 = piece_len[:,0]

        if run ==0: rangeG = np.arange(G)[:3*G//4]
        elif run == subruns-1: rangeG=np.arange(G)[-3*G//4:]
        else: rangeG = np.arange(G)[G//4:3*G//4]

        temp_piece = np.zeros(G, dtype=int)

        for j in range(ntip_y[0]+1):

           for g in range(G):
            temp_piece[g] = piece0[g]
            if g==0:
              for i in range( left_grains[run], left_grains[run]+temp_piece[g]):
                if (i>nx-1 or j>ny-1): break
                angle_field[i,j] = angles[aseq[g]]

            else:
              for i in range(left_grains[run]+temp_piece[g-1], left_grains[run]+temp_piece[g]):
                if (i>nx-1 or j>ny-1): break         
                angle_field[i,j] = angles[aseq[g]]


  #=========================start fill the final field=================

        temp_piece = np.zeros((G, ny), dtype=int)
        extra_y = 0
        if extra_time>0: extra_y = int(extra_time*( ntip_y[final] - ntip_y[final-1]))
        y_range = np.arange(ntip_y[0]+1, ntip_y[final-1]+extra_y+1)

        for g in range(G):
          fint = interp1d(ntip_y, piece_len[g,:],kind='linear')
          new_f = fint(y_range)
          temp_piece[g,y_range] = np.asarray(np.round(new_f),dtype=int)


        for j in y_range:

           #print(temp_piece)
           #temp_piece = np.asarray(np.round(temp_piece/np.sum(temp_piece)*nx),dtype=int)
           for g in rangeG:
            if g==0:
              for i in range(left_grains[run], left_grains[run]+temp_piece[g,j]):
                if (i>nx-1 or j>ny-1): break

                angle_field[i,j] = angles[aseq[g]]

            else:
              for i in range(left_grains[run]+temp_piece[g-1,j], left_grains[run]+temp_piece[g,j]):
                if (i>nx-1 or j>ny-1): break
        
                angle_field[i,j] = angles[aseq[g]]



    #=========================start fill the extra field=================
        area = area_a[run]
        y_f = y_range[-1]
        for g in rangeG:

          top_layer = temp_piece[g, y_f]- temp_piece[g-1, y_f] if g>0 else temp_piece[0, y_f]#p_len[g, final-1]
          if  top_layer==0: height =0 
          else: height = int(np.round((area[g]/top_layer)))
          #print(height)
          for j in range(y_f, y_f+height):

            if g==0:
              for i in range(left_grains[run], left_grains[run]+temp_piece[g, y_f]):
                if (i>nx-1 or j>ny-1): break
                angle_field[i,j] = angles[aseq[g]]

            else:
              for i in range(left_grains[run]+temp_piece[g-1, y_f], left_grains[run]+temp_piece[g, y_f]):
                if (i>nx-1 or j>ny-1): break         
                angle_field[i,j] = angles[aseq[g]]

    #=========================fill in =================

      for j in y_range:

         zeros = np.arange(nx)[angle_field[:,j]<1e-5]
         nonzeros = np.arange(nx)[angle_field[:,j]>=1e-5]

         fint = interp1d(nonzeros, angle_field[nonzeros,j],kind='nearest',fill_value='extrapolate')
         angle_field[zeros,j] = fint(zeros)

  #========================start plotting area, plot ini_field, alpha_true, and field======
      angle_field[:,:ntip_y[0]+1] = alpha_true[:,:ntip_y[0]+1]
    y_top = next( i for i,x  in  enumerate(np.mean(alpha_true, axis=0)) if x<1e-5)
    miss_rate = np.sum( np.absolute(alpha_true[:,:y_top]-angle_field[:,:y_top])>1 )/(nx*y_top)
   # error=field-alpha_true[1:-1,1:-1]
    plot_flag=True
    if plot_flag==True:
      fig = plt.figure(figsize=(7.5*xmax/140,30*ytop/140))
      txt = r'$\epsilon_k$'+str(anis)+'_G'+str("%1.1f"%G0)+r'_$R_{max}$'+str(Rmax)
     # fig.text(.5, .2, txt, ha='center')
      print(alpha_true)
      print(angle_field) 
      ax3 = fig.add_subplot(311)
      cs3 = ax3.imshow(alpha_true.T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      #subplot_rountine(fig, ax3, cs3, 3)
      ax3.yaxis.set_ticks([0,30,60])
      ax3.xaxis.set_ticks([0,50,100,150,200,250,300])
      ax3.set_xlabel(r'$x (\mu m)$')
      ax3.set_ylabel(r'$y (\mu m)$')
      cs3.set_clim(vmin, vmax)
      ax3.set_title('PDE', color=bg_color, fontsize=28)
      trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
      ax3.text(0.5,0.9, r'$t=$'+str("%1.2f"%(60/Rmax*plot_idx/100))+r'$\mu s$', transform=ax3.transAxes + trans, fontsize=ft, horizontalalignment='center')

      ax4 = fig.add_subplot(312)
      cs4 = ax4.imshow(angle_field.T,cmap=newcmp,origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax4, cs4, 3)
      ax4.yaxis.set_ticks([0,30,60])
      ax4.xaxis.set_ticks([0,50,100,150,200,250,300])
      ax4.set_title('GrainNN', color=bg_color, fontsize=28)
      
      ax5 = fig.add_subplot(313)
      cs5 = ax5.imshow( 1* (np.absolute(alpha_true-angle_field)>1 ).T,cmap='Reds',origin='lower',extent= (xmin,xmax, ymin, ytop))
      subplot_rountine(fig, ax5, cs5, 4)
      ax5.yaxis.set_ticks([0,30,60])
      ax5.xaxis.set_ticks([0,50,100,150,200,250,300])
      ax5.set_title('Error '+str(int(np.round(miss_rate*100)))+'%',color=bg_color,fontsize=28)


      plt.savefig(var + '_grains' + str(G) + '_frame' + f'{plot_idx:03}'+'.png',dpi=200,facecolor="white", bbox_inches='tight')
 
      plt.close()


