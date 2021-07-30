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
from torch.utils.data import Dataset, DataLoader
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
num_runs = 900
total_size = frames*num_runs
seq = 1
G = 8     # G is the number of grains
param_len = 0
time_tag = 1
param_list = ['anis','G0','Rmax']
input_len = 2*G + param_len + time_tag
hidden_dim = 40
output_len = G
LSTM_layer = 1
valid_ratio = 0.1

num_train = int((1-valid_ratio)*num_runs)
num_test = num_runs-num_train
window = 10
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

frac_train_ini = frac_train[:,0,:]
frac_test_ini = frac_test[:,0,:]
## subtract the initial part of the sequence, so we can focus on the change
frac_train = frac_train - frac_train_ini[:,np.newaxis,:]
frac_test = frac_test - frac_test_ini[:,np.newaxis,:]


class PrepareData(Dataset):

     def __init__(self, input_, output_, init):
          if not torch.is_tensor(input_):
              self.input_ = torch.from_numpy(input_)
          if not torch.is_tensor(output_):
              self.output_ = torch.from_numpy(output_)
          if not torch.is_tensor(init):
              self.init = torch.from_numpy(init)
     def __len__(self):
         #print('len of the dataset',len(self.output_[:,0]))
         return len(self.output_[:,0])
     def __getitem__(self, idx):
         return self.input_[idx,:], self.output_[idx,:], self.init[idx,:]

# Shape the inputs and outputs
input_seq = np.zeros(shape=(num_train*(frames-window),input_len))
output_seq = np.zeros(shape=(num_train*(frames-window),output_len))
input_test = np.zeros(shape=(num_test*(frames-window),input_len))
output_test = np.zeros(shape=(num_test*(frames-window),output_len))
ini_train = np.zeros(shape=(num_train*(frames-window),output_len))
ini_test = np.zeros(shape=(num_test*(frames-window),output_len))
# Setting up inputs and outputs
sample = 0
for run in range(num_train):
    lstm_snapshot = frac_train[run,:,:]
    for t in range(window,frames):
        input_seq[sample,:output_len] = lstm_snapshot[t-window:t,:]
        input_seq[sample,output_len:-1] = param_train[run,:]
        input_seq[sample,-1] = t/(frames-1) 
        output_seq[sample,:] = lstm_snapshot[t,:]
        ini_train[sample,:] = frac_train_ini[run,:]
        sample = sample + 1

sample = 0
for run in range(num_test):
    lstm_snapshot = frac_test[run,:,:]
    for t in range(window,frames):
        input_test[sample,:output_len] = lstm_snapshot[t-window:t,:]
        input_test[sample,output_len:-1] = param_test[run,:]
        input_test[sample,-1] = t/(frames-1) 
        output_test[sample,:] = lstm_snapshot[t,:]
        ini_test[sample,:] = frac_test_ini[run,:]
        sample = sample + 1
        
input_dat = torch.from_numpy(input_seq)
input_test_pt = torch.from_numpy(input_test)
output_dat = torch.from_numpy(output_seq)
output_test_pt = torch.from_numpy(output_test)

train_loader = PrepareData(input_seq, output_seq, ini_train)
test_loader = PrepareData(input_test, output_test, ini_test)
train_loader = DataLoader(train_loader, batch_size = 256, shuffle=True)
test_loader = DataLoader(test_loader, batch_size = num_test*(frames-window), shuffle=True)

#input_dat = input_dat.permute(1,0,2)
#input_test_pt = input_test_pt.permute(1,0,2)

# train
class LSTM_soft(nn.Module):
    def __init__(self,input_len,output_len,hidden_dim,num_layer):
        super(LSTM_soft, self).__init__()
        self.input_len = input_len
        self.output_len = output_len  
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lstm = nn.GRU(input_len,hidden_dim,num_layer,batch_first=True)
        self.project = nn.Linear(hidden_dim, output_len) # input = [batch, dim] 
        #self.frac_ini = frac_ini
    def forward(self, input_frac, frac_ini):
        
        lstm_out, _ = self.lstm(input_frac)
        target = self.project(lstm_out[:,-1,:])
       # frac = F.softmax(target,dim=1) # dim0 is the batch, dim1 is the vector
        target = F.relu(target+frac_ini)
        frac = F.normalize(target,p=1,dim=1)-frac_ini # dim0 is the batch, dim1 is the vector
        return frac

def LSTM_train(model, num_epochs, train_loader, test_loader):
    
    learning_rate=1e-2
    #torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate) 
                              #   weight_decay=1e-5) # <--
  #  outputs = []
    for epoch in range(num_epochs):

      for  ix, (I_train, O_train, ini_train) in enumerate(train_loader):   

         #print(I_train.shape)
         recon = model(I_train,ini_train)
         loss = criterion(recon, O_train)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
       # optimizer.zero_grad() 
      for  ix, (I_test, O_test, ini_test) in enumerate(test_loader):
        pred = model(I_test,ini_test)
        test_loss = criterion(pred, O_test)
        #print(recon.shape,O_train.shape,pred.shape, O_test.shape)
        print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(loss), float(test_loss)))
        # outputs.append((epoch, data, recon),)
        
    return model 


model = LSTM_soft(input_len, output_len, hidden_dim, LSTM_layer)
model = model.double()

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)
num_epochs = 200
model=LSTM_train(model, num_epochs, train_loader, test_loader)


torch.save(model.state_dict(), './lstmmodel')



## plot to check if the construction is reasonable
# pick a random 
plot_idx = 4  # in test dataset
frame_id = idx[plot_idx+num_train]

filename = filebase+str(frame_id)+ '_rank0.h5'
ft = h5py.File(filename, 'r')
alpha_true = np.asarray(ft['alpha'])
aseq_test = np.asarray(ft['sequence'])
tip_y = np.asarray(ft['y_t'])

pred_frames= frames-window

# evole physics based on trained network
evolve_runs = 1
frac_out_info = frac_test[plot_idx,:window,:]
frac_out = np.zeros((frames,G))
frac_out[:window,:] = frac_out_info[:,:]
train_dat = np.zeros((evolve_runs,input_len))
train_dat[0,output_len:-1] = param_test[plot_idx,:]
print('seq', param_test[plot_idx,:])
frac_out_true = output_test_pt.detach().numpy()[plot_idx*pred_frames:(plot_idx+1)*pred_frames,:]
for i in range(pred_frames):
    train_dat[0,:output_len] = frac_out_info
    train_dat[0,-1] = (i+window)/(frames-1) 
    #print(train_dat.shape)
   # train_dat = torch.from_numpy(np.vstack(frac_out_info,))
    frac_new_vec = model(torch.from_numpy(train_dat).permute(1,0,2), torch.from_numpy(frac_test_ini[[plot_idx],:])).detach().numpy() 
    print('timestep ',i)
    print('predict',frac_new_vec)
    print('true',frac_out_true[i,:])
    frac_out[window+i] = frac_new_vec
    #print(frac_new_vec)
    frac_out_info[:,:] = frac_new_vec
    
frac_out = frac_out + frac_test_ini[[plot_idx],:]
#frac_out_trained = output_test_pt.detach().numpy()[plot_idx*pred_frames:(plot_idx+1)*pred_frames,:]
#frac_out = np.vstack((frac_out_info, frac_out_trained))
#print(frac_out)
print('plot_id,run_id', plot_idx,frame_id)
#print(frac_out_trained)
plot_real(x,y,alpha_true)
plot_reconst(G,x,y,aseq_test,tip_y,alpha_true,frac_out.T)






