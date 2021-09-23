#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:12:11 2021

@author: yigongqin
"""


batch = 1100
num_batch = 1
num_runs = batch*num_batch
valid_ratio = 1/11
num_train_all = int((1-valid_ratio)*num_runs)
num_test = num_runs-num_train_all
num_train = num_batch*100 #num_train_all
num_train_b = int(num_train_all/num_batch)
num_test_b = int(num_test/num_batch)

window = 5
frames = 26
pred_frames= frames-window
total_size = frames*num_runs

G = 8     # G is the number of grains
param_len = 1   # how many parameters
time_tag = 1
input_len = 2*G + param_len + time_tag
hidden_dim = 50
output_len = G
LSTM_layer = 4

num_epochs = 80
learning_rate=5e-4
expand = 10 

seed = 1   