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

window = 5
out_win = 3
frames = 26
pred_frames= frames-window
sam_per_run = frames - window - (out_win-1)
total_size = frames*num_runs


G = 8     # G is the number of grains
param_len = 2   # how many parameters
time_tag = 1
input_len = 2*G + param_len + time_tag
output_len = G

## architecture
hidden_dim = 64
LSTM_layer = 2

num_epochs = 40
learning_rate=5e-4
expand = 10 

seed = 1   
#data_dir = '../../G_E/*'
data_dir = '../../G_E/*'
skip_check = False

