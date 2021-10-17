#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:12:11 2021

@author: yigongqin
"""


batch = 20
#batch = 200
num_batch = 16
#num_batch = 4 

num_runs = batch*num_batch
valid_ratio = 1
num_train_all = int((1-valid_ratio)*num_runs)
num_test = num_runs-num_train_all
num_train = num_train_all
num_train_b = int(num_train_all/num_batch)
num_test_b = int(num_test/num_batch)

window = 5
out_win = 3
frames = 26
train_frames = frames
pred_frames= frames-window
sam_per_run = frames - window - (out_win-1)
total_size = frames*num_runs
dt = 1.0/(frames-1)

G = 8     # G is the number of grains
param_len = G + 3   # how many parameters, color plus 3 physical
output_len = G

## architecture
hidden_dim = 32
LSTM_layer = 3
kernel_size = (3,)

num_epochs = 40
learning_rate=5e-4
area_scale = 0.1

seed = 1   
#data_dir = '../../G_E_auto/*'
#data_dir = '../../G_E_test/*'
data_dir = '../../distribution/*'
skip_check = True

