#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:12:11 2021

@author: yigongqin
"""

batch = 220
num_batch = 20
num_runs = batch*num_batch
valid_ratio = 1/11
num_train_all = int((1-valid_ratio)*num_runs)
num_test = num_runs-num_train_all
num_train = num_batch*100 #num_train_all
num_train_b = int(num_train_all/num_batch)
num_test_b = int(num_test/num_batch)

window = 1
out_win = 4
frames = 26
train_frames = window+out_win
pred_frames = out_win
sam_per_run = 1
total_size = frames*num_runs
dt = 1.0/(frames-1)

G = 8     # G is the number of grains
param_len = G + 3   # how many parameters, color plus 3 physical
output_len = G

## architecture
hidden_dim = 32
LSTM_layer = 3
kernel_size = (3,)

num_epochs = 60
learning_rate= 100e-4
area_scale = 0.1

seed = 1   
#data_dir = '../../G_E/*'
data_dir = '../../G_E/*'
skip_check = False
