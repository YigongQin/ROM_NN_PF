#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:12:11 2021

@author: yigongqin
"""

batch = 1
num_batch = 1800
num_runs = batch*num_batch
#valid_ratio = 1
#num_train_all = int((1-valid_ratio)*num_runs)
num_test = 100 #num_runs-num_train_all
num_train = num_batch*1 #num_train_all
num_train_b = 0 #int(num_train_all/num_batch)
#num_test_b = int(num_test/num_batch)

window = 1
out_win = 4
all_frames = 601
frames = 25
train_frames = window+out_win
pred_frames = out_win
sam_per_run = frames - window - (out_win-1) 
total_size = frames*num_runs
dt = 1.0/(frames-1)

G = 8     # G is the number of grains
G_small = 8
param_len = G + 4   # how many parameters, color plus 3 physical
output_len = G

## architecture
hidden_dim = 16
LSTM_layer = (4, 4)
LSTM_layer_ini = (4,4)
kernel_size = (3,)

num_epochs = 60
learning_rate=100e-4
area_scale = 0.1
size_scale = 1
seed = 1   
data_dir = '../../twoD/*.h5'
#data_dir = '../../ML_PF8_train1000_test100_Mt24002_grains8_frames25_anis0.050_G050.000_Rmax1.000_seed0_rank0.h5'
valid_dir = '../../validation/*.h5'
#valid_dir = data_dir
skip_check = False
