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
    size_b = seq_dat.shape[0]
    size_t = seq_dat.shape[1]
    size_v = seq_dat.shape[2]

    size_p = param_dat.shape[1]


    Gi = np.arange(G)
    Pi = np.arange(size_p-2*G_all)

    new_size_v = size_v -   G_all + G
    new_size_p = size_p - 2*G_all + 2*G
    
    if G==G_all: 
        return param_dat, seq_dat, 1
          
        
    elif G_all>G:
        expand = (G_all-G-2)//2 + 2
        new_param = np.zeros((expand*size_b, new_size_p))
        new_seq = np.zeros((expand*size_b, size_t, new_size_v))

        for i in range(expand):
            
            slice_param = np.concatenate(( Gi+2*i, Gi+2*i+G_all, Pi+2*G_all ))
            slice_seq = np.concatenate(( Gi+2*i, size_v-1))

            new_param[i*size_b:(i+1)*size_b,:] = param_dat[:,slice_param]
            new_seq[i*size_b:(i+1)*size_b,:,:] = seq_dat[:,slice_seq]
        
        return new_param, new_seq, expand
            
    else: raise ValueError("number of grain is wrong")



def merge_grain(seq_dat, G, G_all, expand):
    
    
    '''
    Assume G and G_all here are all even numbers 
    G = N_w +2, N_w is the no. grains one grain can affect
    '''

    size_b = seq_dat.shape[0]
    size_t = seq_dat.shape[1]
    size_v = seq_dat.shape[2]
    frac = seq_dat[:,:,:-1]
    y = seq_dat[:,:,-1]

    assert size_b%expand == 0
    new_size_b = size_b//expand

    mid = np.array([G//2-1, G//2])
    BC_l = G//2+1

    new_size_v = size_v +  G_all - G
    
    if G==G_all: 
        return frac, y
          
        
    elif G_all>G:


        new_frac = np.zeros((new_size_b, size_t, new_size_v-1))

        ## first give the first and last data
        new_frac[:,:,:BC_l]  = frac[:new_size_b, :,:BC_l]
        new_frac[:,:,-BC_l:] = frac[-new_size_b:,:,-BC_l:]

        ## add the two middle grains to the data
        for i in range(1, expand-1):
 
            new_frac[:,:,BC_l+2*i-2:BC_l+2*i] = frac[new_size_b*i:new_size_b*(i+1),:,mid]
        
        return new_frac, y
            
    else: raise ValueError("number of grain is wrong")    



