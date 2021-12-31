#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:03:38 2021

@author: yigongqin
"""
import numpy as np


def find_weird(frac_train, thre):

    
    diff_arr = np.absolute(np.diff(frac_train,axis=1))
    print(diff_arr.shape)
    print('maximum abs change',np.max(diff_arr))
    weird_p = np.where(diff_arr>thre)
    print('where the points are',weird_p)
    weird_sim = weird_p[0]
    weird_sim = list(set(list(weird_sim))) 
    print('weird values',diff_arr[np.where(diff_arr>thre)])
    diff_arr[diff_arr==0.0]=np.nan
    print('the mean of the difference',np.nanmean(diff_arr))
    print('number of weird sim',len(weird_sim)) 
      
    return weird_sim

def redo_divide(frac_train, weird_sim, param_train, G, frames):
    # frac_train shape [frames,G]
    
   # for sid in [54]:
    for sid in weird_sim:
        ## redo the process of the 
      frac = frac_train[sid,:,:].squeeze()
      aseq = param_train[sid,:G].squeeze()*4.5+5.5
      #if sid ==54:
      #  print('weird sim ',sid ,'before',frac)
      #  print(aseq)
      left_coor = np.cumsum(frac[0,:])-frac[0,:]
      #print('left_coor',left_coor)
      for kt in range(1,frames):
        for j in range(1,G):
          if frac[kt,j]<1e-4 and frac[kt-1,j]>1e-4:
            left_nozero = j-1;
            while left_nozero>=0: 
                if frac[kt,left_nozero]>1e-4: break
                else: left_nozero-=1
            if left_nozero>=0 and aseq[left_nozero]==aseq[j]:
               #print("find sudden merging\n");
               all_piece = frac[kt,left_nozero]
               pre_piece = left_coor[j] - left_coor[left_nozero] 
               if pre_piece<0: pre_piece=0
               if pre_piece>all_piece: pre_piece=all_piece
               cur_piece = all_piece - pre_piece
               frac_train[sid,kt,left_nozero] = pre_piece
               frac_train[sid,kt,j] = cur_piece
               #print("correction happens, %d grain frac %f, %d grain frac %f\n" %(left_nozero,frac_train[sid,kt,left_nozero],j,frac_train[sid,kt,j]))
                      
          else:
            if j>0: left_coor[j] = (np.cumsum(frac[kt,:])-frac[kt,:])[j]
            

def check_data_quality(frac_all,param_all,y_all,G,frames):

    ### C1 check the fraction jump
    weird_sim = find_weird(frac_all, 0.15)
    refine_count=0
    while len(weird_sim)>0:
        refine_count  +=1
        redo_divide(frac_all, weird_sim, param_all, G, frames)
        weird_sim = find_weird(frac_all, 0.15)
        if refine_count ==5: break
    print(weird_sim)
    #print(param_all[weird_sim,-2:])
    ### C2 go to zero but emerge again 
    
    merge_arg = np.where( (frac_all[:,:-1,:]<1e-4)*1*(frac_all[:,1:,:]>1e-4) )
    print("renaissance", np.sum( (frac_all[:,:-1,:]<1e-4)*1*(frac_all[:,1:,:]>1e-4) ) )
    print("renaissance points", merge_arg)
    weird_sim = weird_sim + list(set(list(merge_arg[0])))
    #print("how emerge", frac_all[:,:-1,:][merge_arg], frac_all[:,1:,:][merge_arg])
    #print(frac_all[2489,:,:])
    
    ### C3 max and min
    print('min and max of training data', np.min(frac_all), np.max(frac_all))
    
    ## #C4 normalization
    diff_to_1 = np.absolute(np.sum(frac_all,axis=2)-1)
    #print(np.where(diff_to_1>1e-4))
    print('max diff from 1',np.max(diff_to_1))
    print('all the summation of grain fractions are 1', np.sum(diff_to_1))
    frac_all /= np.sum(frac_all,axis=2)[:,:,np.newaxis] 
    diff_to_1 = np.absolute(np.sum(frac_all,axis=2)-1)
    print('all the summation of grain fractions are 1', np.sum(diff_to_1))
    
    ### C5 check y_all for small values
    last_y = y_all[:,-1]
    print('mean and std of last y',np.mean(last_y),np.std(last_y))
    weird_y_loc = np.where(last_y<np.mean(last_y)-6*np.std(last_y))
    print('where they is small ', weird_y_loc)
    print('weird values ', last_y[weird_y_loc])
    print('weird y traj',y_all[weird_y_loc,:])
    weird_sim = weird_sim + list(weird_y_loc[0])
    return list(set(weird_sim))
    
    
    
