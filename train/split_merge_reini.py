#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""

import numpy as np


def list_subtract(family, child):

    for x in child:
       family.remove(x)

    return family


def map_grain(frac_layer, G, G_all):
      

    '''

    return a variable length array

    '''

    assert len(frac_layer)==G_all
    min_pixels = 2   ## at least have two pixels
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
        return np.array( joint_list )


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


def split_grain(param_dat, seq_dat, G, G_all):
    
    
    '''
    Assume G and G_all here are all even numbers 
    G = N_w +2, N_w is the no. grains one grain can affect
    '''

    size_b = seq_dat.shape[0]
    size_t = seq_dat.shape[1]
    size_v = seq_dat.shape[2]  # should be 3*G_all+1

    size_p = param_dat.shape[1] # should be 2G_all+4


  #  Gi = np.arange(G)
    Pi = np.arange(size_p-2*G_all)

    new_size_v = size_v - 3*G_all + 3*G
    new_size_p = size_p - 2*G_all + 2*G
    
    if G==G_all: 
        return param_dat, seq_dat, [np.arange(G)], np.ones((size_b,1)), np.zeros((size_b,1))
         
        
    elif G_all>G:


        grain_arg_list = []  ## a list of variable-size array [num_subruns, G]

        for run in range(size_b):

          frac_layer = seq_dat[run,-1,:G_all]  ## use the last time step to resplit

          args = map_grain(frac_layer, G, G_all)

          grain_arg_list.append(args)

          for i in range(args.shape[0]):  ## every i is a subsimulation
            
            grain_id = args[i,:]

            ## ============== scaling region ============= ##

            #print(seq_dat[run,:,list(grain_id)].shape)
            df = np.sum( seq_dat[run][:,grain_id], axis = -1 )  ## not sure what to do with it, just 1d with one number

            df_loc = df*(G_all/G)  ##should be close enough to 1

            param_sliced = param_dat[run][grain_id]/df  ## initial
           # print(param_sliced, param_dat[run,grain_id], df)
            frac_sliced =  seq_dat[run][:,grain_id]/df[:,np.newaxis]  ## frac
           # print(frac_sliced, seq_dat[run][:,grain_id], df)
            dfrac_sliced = seq_dat[run][:,grain_id+G_all]/df[:,np.newaxis]  # dfrac

            darea_sliced = seq_dat[run][:,grain_id+2*G_all]/ df_loc[:,np.newaxis] 
            

       #     if i>(expand-1)//2: left_coors[:,i] = G_all/G*(1- np.cumsum(seq_dat[:,0,:], axis=-1)[:,G+2*i-1]) 
       #     elif i>0: left_coors[:,i] = G_all/G*np.cumsum(seq_dat[:,0,:], axis=-1)[:,2*i-1]
        #    else: pass
  
            assert np.linalg.norm( np.sum(param_sliced,axis=-1) - 1 ) <1e-5
            assert np.linalg.norm( np.sum(frac_sliced,axis=-1) - np.ones(size_t) ) <1e-5
            assert np.linalg.norm( np.sum(dfrac_sliced,axis=-1) - np.zeros(size_t) ) <1e-5


           ## ============== scaling region ============= ##

           # new_param[i*size_b:(i+1)*size_b,:G] = param_sliced
            slice_param = np.concatenate(( grain_id+G_all, Pi+2*G_all ))

           # new_param[i*size_b:(i+1)*size_b,:] = param_dat[:,slice_param]
           # new_seq[i*size_b:(i+1)*size_b,:,-1] = seq_dat[:,:,-1]


           # new_seq[i*size_b:(i+1)*size_b,:,:G]  = frac_sliced
           # new_seq[i*size_b:(i+1)*size_b,:,G:2*G] = dfrac_sliced
           # new_seq[i*size_b:(i+1)*size_b,:,2*G:3*G] = darea_sliced


            seq_1 = np.concatenate(( frac_sliced, dfrac_sliced, darea_sliced, seq_dat[run][:,-1:]), axis = -1)
            param_1 = np.concatenate(( param_sliced, param_dat[run,slice_param]), axis = -1)

            if run==0 and i==0:

                new_seq = seq_1[np.newaxis,:,:]
                new_param = param_1[np.newaxis,:]
                domain_factor = df_loc[np.newaxis,:]

            else:
   
                new_seq = np.concatenate((new_seq, seq_1[np.newaxis,:,:]), axis=0)
                new_param = np.concatenate((new_param, param_1[np.newaxis,:]), axis=0)
                domain_factor = np.concatenate((domain_factor, df_loc[np.newaxis,:]), axis=0)



        return new_param, new_seq, grain_arg_list, domain_factor, np.zeros((new_seq.shape[0],1))
            
    else: raise ValueError("number of grain is wrong")



def merge_grain(frac, y, area, G, G_all, grain_arg_list, domain_factor, left_coors):
    
    
    '''
    Assume G and G_all here are all even numbers 
    G = N_w +2, N_w is the no. grains one grain can affect
    '''

    size_b = frac.shape[0]
    size_t = frac.shape[1]
    size_v = frac.shape[2]



    #assert size_b%expand == 0
    new_size_b = len(grain_arg_list)  ## number of real simulation 

    BC_l = G//2+1

    new_size_v = size_v +  G_all - G
    



    if G==G_all: 
        return frac, y, area, np.cumsum(frac, axis=-1) - frac
          
        
    elif G_all>G:

        increment = 0

        
        new_frac = np.zeros((new_size_b, size_t, new_size_v))
        new_area = np.zeros((new_size_b, size_t, new_size_v))
        new_y = np.zeros((new_size_b, size_t))
        std_y = np.zeros((new_size_b, size_t))

        for run in range(new_size_b):


            args = grain_arg_list[run]
            expand = args.shape[0]
            ## =========== the y part =================

            

            y_null = y[increment:increment+expand,:]

            new_y[run,:] = np.mean(y_null, axis = 0)

            std_y[run,:] = np.std(y_null, axis = 0)
           # new_y = np.min(y_null, axis = 0)

            ## =========== the y part =================



            ## add the two middle grains to the data
            for i in range(expand):

                subruns = increment + i
            
                if i==0:
                    new_frac[run][:,args[i,:BC_l]]  = frac[subruns, :,:BC_l]*domain_factor[subruns,:,np.newaxis]*G/G_all
                    new_area[run][:,args[i,:BC_l]]  = area[subruns, :,:BC_l]*domain_factor[subruns,:,np.newaxis]
     
                if i==expand-1:
                    new_frac[run][:,args[i,-BC_l:]] = frac[subruns,:,-BC_l:]*domain_factor[subruns,:,np.newaxis]*G/G_all
                    new_area[run][:,args[i,-BC_l:]] = area[subruns,:,-BC_l:]*domain_factor[subruns,:,np.newaxis] 
      
                if i>0 and i<expand-1:
                    new_frac[run][:,args[i,G//2-1:G//2+1]] = frac[subruns,:,G//2-1:G//2+1]*domain_factor[subruns,:,np.newaxis]*G/G_all
                    new_area[run][:,args[i,G//2-1:G//2+1]] = area[subruns,:,G//2-1:G//2+1]*domain_factor[subruns,:,np.newaxis] 

            increment += expand

        #new_frac *= G/G_all
       # left_coors_grains *= G/G_all
        ## evaluation (a) sum frac, (b) std of y
        diff_1 = np.absolute( np.sum(new_frac,axis=-1) - np.ones_like(new_y)  )
        max_1 = np.max( diff_1 ); mean_1 = np.mean( diff_1)
        max_y = np.max( std_y )

        print('evaluate split-merge grain strategy', max_1, mean_1, max_y)

      #  assert left_coors_grains.shape[2]==new_frac.shape[2]
        return new_frac, new_y, new_area, np.zeros((new_size_b, size_t, G_all))
            
    else: raise ValueError("number of grain is wrong")    



