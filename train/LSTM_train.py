#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_funcs import plot_reconst,plot_real
#rnn = nn.GRU(10, 20, 2)
#inputs = torch.randn(5, 3, 10)
#h0 = torch.randn(2, 3, 20)
#outputg, hn = rnn(inputs, h0)

#rnn = nn.LSTM(10, 1, 1)  # dimension input*output*layer
#inputs = torch.randn(5, 1, 10) # dimension sequence * batch size * vector legnth

#output, (hn, cn) = rnn(inputs)
#h0 = torch.randn(2, 3, 20) # layer * batch * output
#c0 = torch.randn(2, 3, 20) # layer * batch * output
#output, (hn, cn) = rnn(inputs, (h0, c0))
#output.size() #torch.Size([5, 3, 20])


#m = nn.Softmax(dim=1)  #dim here is axis
#input_s = torch.randn(2, 3)
#output = m(input_s)


# global parameters

frames = 101
num_runs = 100
total_size = frames*num_runs
seq = 1
G = 8     # G is the number of grains
param_len = 0
time_tag = 1
param_list = ['anis','G0','Rmax']
input_len = 2*G + param_len + time_tag
hidden_dim = 20
output_len = G
LSTM_layer = 1
valid_ratio = 0.1

num_train = int((1-valid_ratio)*num_runs)
num_test = num_runs-num_train
window = 5
seed = 1

# global information that apply for every run
filebase = '../../ML_Mt47024_grains8_anis0.130_seed'
filename = filebase+str(1)+ '_rank0.h5'
f = h5py.File(filename, 'r')
x = np.asarray(f['x_coordinates'])
y = np.asarray(f['y_coordinates'])
xmin = x[1]; xmax = x[-2]
ymin = y[1]; ymax = y[-2]
print('xmin',xmin,'xmax',xmax,'ymin',ymin,'ymax',ymax)
dx = x[1]-x[0]
fnx = len(x); fny = len(y); nx = fnx-2; ny = fny-2;
print('nx,ny', nx,ny)

## =======load data and parameters from the every simulation======
#alpha_true = np.asarray(f['alpha'])
#alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')

#tip_y = np.asarray(f['y_t'])
#ntip_y = np.asarray(tip_y/dx,dtype=int)
#print('ntip',ntip_y)

frac_all = np.zeros((num_runs,frames,G)) #run*frames*vec_len
param_all = np.zeros((num_runs,G+param_len))



for run in range(num_runs):
    filename = filebase+str(run)+ '_rank0.h5'
    f = h5py.File(filename, 'r')
    aseq = np.asarray(f['sequence'])  # 1 to 10
    Color = (aseq-5.5)/4.5        # normalize C to [-1,1]
    #print('angle sequence', Color)
    frac = (np.asarray(f['fractions'])).reshape((G,frames), order='F')  # grains coalese, include frames
    frac = frac.T
    frac_all[run,:,:] = frac
    param_all[run,:G] = Color


# trained dataset need to be randomly selected:
idx =  np.arange(num_runs)
np.random.seed(seed)
np.random.shuffle(idx)

frac_train = frac_all[idx[:num_train],:,:]
frac_test = frac_all[idx[num_train:],:,:]
param_train = param_all[idx[:num_train],:]
param_test = param_all[idx[num_train:],:]

# Shape the inputs and outputs
input_seq = np.zeros(shape=(num_train*(frames-window),window,input_len))
output_seq = np.zeros(shape=(num_train*(frames-window),output_len))
input_test = np.zeros(shape=(num_test*(frames-window),window,input_len))
output_test = np.zeros(shape=(num_test*(frames-window),output_len))
# Setting up inputs and outputs
sample = 0
for run in range(num_train):
    lstm_snapshot = frac_train[run,:,:]
    for t in range(window,frames):
        input_seq[sample,:,:output_len] = lstm_snapshot[t-window:t,:]
        input_seq[sample,:,output_len:-1] = param_train[run,:]
        input_seq[sample,:,-1] = t/(frames-1) 
        output_seq[sample,:] = lstm_snapshot[t,:]
        sample = sample + 1
sample = 0
for run in range(num_test):
    lstm_snapshot = frac_test[run,:,:]
    for t in range(window,frames):
        input_test[sample,:,:output_len] = lstm_snapshot[t-window:t,:]
        input_test[sample,:,output_len:-1] = param_test[run,:]
        input_test[sample,:,-1] = t/(frames-1) 
        output_test[sample,:] = lstm_snapshot[t,:]
        sample = sample + 1
        
input_dat = torch.from_numpy(input_seq)
input_test_pt = torch.from_numpy(input_test)
output_dat = torch.from_numpy(output_seq)
output_test_pt = torch.from_numpy(output_test)

input_dat = input_dat.permute(1,0,2)
input_test_pt = input_test_pt.permute(1,0,2)

# train
class LSTM_soft(nn.Module):
    def __init__(self,input_len,output_len,hidden_dim,num_layer):
        super(LSTM_soft, self).__init__()
        self.input_len = input_len
        self.output_len = output_len  
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lstm = nn.GRU(input_len,hidden_dim,num_layer)
        self.project = nn.Linear(hidden_dim, output_len) # input = [batch, dim] 
        
    def forward(self, input_frac):
        
        lstm_out, _ = self.lstm(input_frac)
        target = self.project(lstm_out[-1,:,:])
        frac = F.softmax(target,dim=1) # dim0 is the batch, dim1 is the vector
        return frac

def LSTM_train(model, num_epochs, I_train, I_test, O_train, O_test):
    
    learning_rate=5e-3
    #torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
  #  outputs = []
    for epoch in range(num_epochs):

        
        recon = model(I_train)
        loss = criterion(recon, O_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        pred = model(I_test)
        test_loss = criterion(pred, O_test)
        #print(recon.shape,O_train.shape,pred.shape, O_test.shape)
        print('Epoch:{}, Train loss:{:.6f}, Test loss:{:.6f}'.format(epoch+1, float(loss), float(test_loss)))
       # outputs.append((epoch, data, recon),)
        
    return  


model = LSTM_soft(input_len, output_len, hidden_dim, LSTM_layer)
model = model.double()

num_epochs = 500
LSTM_train(model, num_epochs, input_dat, input_test_pt, output_dat, output_test_pt)



## plot to check if the construction is reasonable
# pick a random 
plot_idx = 5  # in test dataset
frame_id = idx[plot_idx+num_train]

filename = filebase+str(frame_id)+ '_rank0.h5'
ft = h5py.File(filename, 'r')
alpha_true = np.asarray(ft['x_coordinates'])
aseq_test = np.asarray(ft['a_seq'])
tip_y = np.asarray(ft['y_t'])

frac_out = output_test_pt[plot_idx,:]
plot_real(x,y,alpha_true)
plot_reconst(G,x,y,aseq_test,tip_y,alpha_true,frac_out)






