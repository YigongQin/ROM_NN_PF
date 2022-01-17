#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""

import numpy as np


def split_grain(param_dat, seq_dat, G, G_all):
    
    
    '''
    Assume G and G_all here are all even numbers 
    G = N_w +2, N_w is the no. grains one grain can affect
    '''
    check_dat = False
    size_b = seq_dat.shape[0]
    size_t = seq_dat.shape[1]
    size_v = seq_dat.shape[2]  # should be 3*G_all+1

    size_p = param_dat.shape[1] # should be 2G_all+4


    Gi = np.arange(G)
    Pi = np.arange(size_p-2*G_all)

    new_size_v = size_v - 3*G_all + 3*G
    new_size_p = size_p - 2*G_all + 2*G
    
    if G==G_all: 
        return param_dat, seq_dat, 1
          
        
    elif G_all>G:
        expand = (G_all-G-2)//2 + 2
        new_param = np.zeros((expand*size_b, new_size_p))
        new_seq = np.zeros((expand*size_b, size_t, new_size_v))

        ones = np.ones((size_b, size_t))
        ones_p = np.ones((size_b))
        zeros = np.zeros((size_b, size_t))

        for i in range(expand):
            
            slice_param = np.concatenate(( Gi+2*i, Gi+2*i+G_all, Pi+2*G_all ))
            #slice_seq = np.concatenate(( Gi+2*i, size_v-1))

            new_param[i*size_b:(i+1)*size_b,:] = param_dat[:,slice_param]
            new_seq[i*size_b:(i+1)*size_b,:,-1] = seq_dat[:,:,-1]

            param_sliced = G_all/G*param_dat[:,2*i:G+2*i]  ## initial
            frac_sliced =  G_all/G*seq_dat[:,:,2*i:G+2*i]  ## frac
            dfrac_sliced = G_all/G*seq_dat[:,:,2*i+G_all:G+2*i+G_all] # dfrac
            darea_sliced = seq_dat[:,:,2*i+2*G_all:G+2*i+2*G_all]

            ## here requires cut/add to make sure unity
            fsum = np.cumsum(frac_sliced, axis=-1)
            psum = np.cumsum(param_sliced, axis=-1)

            
            frac_sliced -= np.diff((fsum>1)*(fsum-1),axis=-1,prepend=0)
         
            param_sliced -= np.diff((psum>1)*(psum-1),axis=-1,prepend=0)

            frac_sliced[:,:,-1] = ones - np.sum(frac_sliced[:,:,:-1], axis=-1)
            dfrac_sliced[:,:,-1] = zeros - np.sum(dfrac_sliced[:,:,:-1], axis=-1)
            param_sliced[:,-1] = ones_p - np.sum(param_sliced[:,:-1], axis=-1)


            assert np.linalg.norm( np.sum(param_sliced,axis=-1) - ones_p ) <1e-5
            assert np.linalg.norm( np.sum(frac_sliced,axis=-1) - ones ) <1e-5
            assert np.linalg.norm( np.sum(dfrac_sliced,axis=-1) - zeros ) <1e-5
            #assert np.all(param_sliced>=0)
            #print(np.where(param_sliced<0))
            new_seq[i*size_b:(i+1)*size_b,:,:G]  = frac_sliced
            new_seq[i*size_b:(i+1)*size_b,:,G:2*G] = dfrac_sliced
            new_seq[i*size_b:(i+1)*size_b,:,2*G:3*G] = darea_sliced
            new_param[i*size_b:(i+1)*size_b,:G] = param_sliced

        if check_dat == True:

            print(seq_dat[0,0,:])
            print(new_seq[::size_b,0,:])
            print(param_dat[0,:])
            print(new_param[::size_b,:])

        return new_param, new_seq, expand
            
    else: raise ValueError("number of grain is wrong")



def merge_grain(frac, y, area, G, G_all, expand):
    
    
    '''
    Assume G and G_all here are all even numbers 
    G = N_w +2, N_w is the no. grains one grain can affect
    '''

    size_b = frac.shape[0]
    size_t = frac.shape[1]
    size_v = frac.shape[2]
    #frac = seq_dat[:,:,:-1]
    #y = seq_dat[:,:,-1]


    assert size_b%expand == 0
    new_size_b = size_b//expand

    mid = np.array([G//2-1, G//2])
    BC_l = G//2+1

    new_size_v = size_v +  G_all - G
    



    if G==G_all: 
        return frac, y, area
          
        
    elif G_all>G:


        new_frac = np.zeros((new_size_b, size_t, new_size_v))
        new_area = np.zeros((new_size_b, size_t, new_size_v))
        ## first give the first and last data
        new_frac[:,:,:BC_l]  = frac[:new_size_b, :,:BC_l]
        new_frac[:,:,-BC_l:] = frac[-new_size_b:,:,-BC_l:]

        new_area[:,:,:BC_l]  = area[:new_size_b, :,:BC_l]
        new_area[:,:,-BC_l:] = area[-new_size_b:,:,-BC_l:]

        y_null = np.zeros((expand, new_size_b, size_t))
        ## add the two middle grains to the data
        for i in range(1, expand-1):
 
            new_frac[:,:,BC_l+2*i-2:BC_l+2*i] = frac[new_size_b*i:new_size_b*(i+1),:,mid]
            new_area[:,:,BC_l+2*i-2:BC_l+2*i] = area[new_size_b*i:new_size_b*(i+1),:,mid]

        new_frac *= G/G_all

        for i in range(expand):

            y_null[i,:,:] = y[new_size_b*i:new_size_b*(i+1),:]

        new_y = np.mean(y_null, axis = 0)
        
        ## evaluation (a) sum frac, (b) std of y
        diff_1 = np.absolute( np.sum(new_frac,axis=-1) - np.ones_like(new_y)  )
        max_1 = np.max( diff_1 ); mean_1 = np.mean( diff_1)
        max_y = np.max( np.std (y_null, axis = 0) )

        print('evaluate split-merge grain strategy', max_1, mean_1, max_y)

        return new_frac, new_y, new_area
            
    else: raise ValueError("number of grain is wrong")    



