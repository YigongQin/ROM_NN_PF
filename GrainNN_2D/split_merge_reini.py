#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""

import numpy as np
from utils import assemb_feat



def list_subtract(family, child):

    for x in child:
       family.remove(x)

    return family


def map_grain(frac_layer, G, G_all):
      

    '''

    return a variable length array

    '''

    assert len(frac_layer)==G_all
    min_pixels = 0   ## at least have two pixels
    pos_arg = np.where( frac_layer>min_pixels/ (400/G*G_all) )[0]  ## numpy array
    pos_arg_list = list(pos_arg)
    num_act_grain = len(pos_arg)
    
    zero_arg_list = list_subtract( list(np.arange(G_all)), pos_arg_list )

   # print(pos_arg_list, zero_arg_list)


    ## num_act_grains range [1, G_all], it can be odd or even. t

    if num_act_grain < G:
        ## only one subrun
        joint_list = sorted( pos_arg_list + zero_arg_list[:(G-num_act_grain)] ) 
        assert len(joint_list) == G
        return np.array( joint_list )[np.newaxis,:]


    elif num_act_grain < G + 2:
        ## two simulations

        return np.stack(( pos_arg[:G], pos_arg[-G:]))


    else: 
        ## general case, 
        ## ***note when pos_arg is odd*** 
        if len(pos_arg)%2==1:
          pos_arg = np.array( sorted( pos_arg_list + zero_arg_list[:1] ) )
        assert len(pos_arg)%2==0
        expand = (len(pos_arg)-G-2)//2 + 2

        args = -np.ones((expand,G), dtype=int)


        for i in range(expand):  ## replace for loop later

            args[i,:] = pos_arg[2*i:G+2*i]

        assert np.all(args>-1)

        return args

def map_grain_fix(frac_layer, G, G_all):
      

        expand = (G_all-G-2)//2 + 2

        args = -np.ones((expand,G), dtype=int)


        for i in range(expand):  ## replace for loop later

            args[i,:] = np.arange(G_all)[2*i:G+2*i]

        assert np.all(args>-1)

        return args

def split_grain(seq_dat, hp):
    
    
    '''

    '''

    size_b = seq_dat.shape[0]
    size_t = seq_dat.shape[1]
    size_c = seq_dat.shape[2] 

    
    if hp.G==hp.G_base: 
        return seq_dat, [np.arange(hp.G)], hp.Cl[:,np.newaxis]
         
        
    elif hp.G>hp.G_base:


        grain_arg_list = []  ## a list of variable-size array [num_subruns, G]

        for run in range(size_b):

            frac_layer = np.mean(seq_dat[run,:,0,:], axis=0)  ## use the last time step to resplit

            args = map_grain(frac_layer, hp.G_base, hp.G)

            grain_arg_list.append(args)

            seq_1b = np.zeros((args.shape[0],size_t,size_c,hp.G_base))
            df_1b = np.zeros((args.shape[0],hp.G_base))

            for i in range(args.shape[0]):  ## every i is a subsimulation
            
                grain_id = args[i,:]

            ## ============== scaling region ============= ##

            #print(seq_dat[run,:,list(grain_id)].shape)
                df = np.sum( seq_dat[run][:,0,grain_id], axis = -1 ).max()[np.newaxis]  ## not sure what to do with it, just 1d with one number
                df_1b[i,:] = df 

                for ch in range(size_c):

                    slice = seq_dat[run][:,ch,grain_id]



                    if ch in [0,1,2,4]:
                        slice /= df[:,np.newaxis]

                        modf_id = 0 if i>(args.shape[0]-1)//2 else -1

                        if ch==1:
                            slice[:,modf_id] += -np.sum(slice, axis=-1)
                            assert np.linalg.norm( np.sum(slice, axis=-1) ) < 1e-5

                        else:
                            slice[:,modf_id] += 1-np.sum(slice, axis=-1)
                            assert np.linalg.norm( np.sum(slice, axis=-1) -1) < 1e-5

                    seq_1b[i,:,ch,:] = slice 

            if run==0 :

                new_seq = seq_1b
                Cl = df_1b

            else:

                new_seq = np.concatenate((new_seq, seq_1b), axis=0)
                Cl = np.concatenate((Cl, df_1b), axis=0)



        return new_seq, grain_arg_list, Cl
            
    else: raise ValueError("number of grain is wrong")


def merge_grain(frac, dseq, hp, grain_arg_list, Cl_list):
    
    
    '''
    update the first four features of seq
    '''

    size_b = frac.shape[0]
    size_t = frac.shape[1]



    if hp.G==hp.G_base: 
        output = np.zeros((size_b,size_t, 4, hp.G))
        assemb_feat(dseq, frac, hp.G, output)
        return output
          
        
    elif hp.G>hp.G_base:

        increment = 0
        dfrac = dseq[:,:,:hp.G_base]
        area = dseq[:,:,hp.G_base:2*hp.G_base]
        y = dseq[:,:,-1]
        
        new_size_b = len(grain_arg_list)  ## number of real simulation 

        BC_l = hp.G_base//2+1

        output = np.zeros((new_size_b, size_t, 4, hp.G))
        std_y = np.zeros((new_size_b, size_t))

        for run in range(new_size_b):


            args = grain_arg_list[run]
            expand = args.shape[0]
            ## =========== the y part =================

            

            y_null = y[increment:increment+expand,:]

            output[run,:,-1,:] = np.mean(y_null, axis = 0)[:,np.newaxis]

            std_y[run,:] = np.std(y_null, axis = 0)
           # new_y = np.min(y_null, axis = 0)

            ## =========== the y part =================

            ## add the two middle grains to the data
            for i in range(expand):

                subruns = increment + i
              
                Cl = Cl_list[subruns,[0],np.newaxis]
            
                if i==0:
                    real_loc = np.arange(BC_l)
                elif i==expand-1:
                    real_loc = np.arange(hp.G_base-BC_l, hp.G_base)
                else:
                    real_loc = np.array([hp.G_base//2-1, hp.G_base//2])
                real_glob = args[i,real_loc]

                output[run,:,0, real_glob] = frac[subruns,:,real_loc]*Cl
                output[run,:,1, real_glob] = dfrac[subruns,:,real_loc]*Cl
                output[run,:,2, real_glob] = area[subruns,:,real_loc]*Cl

            increment += expand



        ## ============= normalization step =============


        output[:,:,0,:] /= hp.G_base/hp.G*np.sum(output[:,:,0,:], axis=-1)[:,:,np.newaxis] 


        max_y = np.max( std_y )

        print('the standard deviation of y interface', max_y)

        return output
            
    else: raise ValueError("number of grain is wrong")    



