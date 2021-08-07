#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""
import time
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
from adabound import *
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
host='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device=host
print('device',device)
model_exist = True
frames = 26
num_runs = 1100
total_size = frames*num_runs
seq = 1
G = 8     # G is the number of grains
param_len = 0
time_tag = 1
param_list = ['anis','G0','Rmax']
input_len = 2*G + param_len + time_tag
hidden_dim = 50
output_len = G
LSTM_layer = 3
valid_ratio = 1/11

num_train = int((1-valid_ratio)*num_runs)
num_test = num_runs-num_train
window = 5
seed = 1
pred_frames= frames-window
print('train, test', num_train, num_test)
print('frames, window', frames, window)

num_epochs = 80
learning_rate=5e-4
expand = 10 #9

# global information that apply for every run
filebase = '../../ML_PF10_train1000_test100_Mt47024_grains8_frames25_anis0.130_seed'
filename = filebase+str(2)+ '_rank0.h5'
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

aseq_asse = np.asarray(f['sequence'])
frac_asse = np.asarray(f['fractions'])
tip_y_asse = np.asarray(f['y_t'])
for run in range(num_runs):
    aseq = aseq_asse[run*G:(run+1)*G]  # 1 to 10
#    Color = (aseq-3)/2        # normalize C to [-1,1]
    Color = (aseq-5.5)/4.5
    #print('angle sequence', Color)
    frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
    frac = frac.T
    frac_all[run,:,:] = frac
    param_all[run,:G] = Color

#print(tip_y_asse[frames::frames])
# trained dataset need to be randomly selected:
idx =  np.arange(num_runs)
np.random.seed(seed)
#np.random.shuffle(idx[:-1])
print(idx)
frac_train = frac_all[idx[:num_train],:,:]
frac_test = frac_all[idx[num_train:],:,:]
param_train = param_all[idx[:num_train],:]
param_test = param_all[idx[num_train:],:]

frac_train_ini = frac_train[:,0,:]
frac_test_ini = frac_test[:,0,:]

#print(frac_train_ini)
print('max and min of training data', np.min(frac_train), np.max(frac_train))
## subtract the initial part of the sequence, so we can focus on the change
frac_train = frac_train - frac_train_ini[:,np.newaxis,:]
frac_test = frac_test - frac_test_ini[:,np.newaxis,:]

## scale the frac according to the time frame 
linear_fact = expand/pred_frames 
scaler_lstm = (frames-np.arange(frames))*linear_fact
#scaler_lstm = np.ones(frames)
frac_train = frac_train* scaler_lstm[np.newaxis,:,np.newaxis]
frac_test = frac_test* scaler_lstm[np.newaxis,:,np.newaxis]

#print('min and max of scaled data', np.min(frac_train,axis=0), np.max(frac_train,axis=0))

def todevice(data):

    return torch.from_numpy(data).to(device)
def tohost(data):

    return data.detach().to(host).numpy()

class PrepareData(Dataset):

     def __init__(self, input_, output_, init, scaler):
          if not torch.is_tensor(input_):
              self.input_ = todevice(input_)
          if not torch.is_tensor(output_):
              self.output_ = todevice(output_)
          if not torch.is_tensor(init):
              self.init = todevice(init)
          if not torch.is_tensor(scaler):
              self.scaler = todevice(scaler)
     def __len__(self):
         #print('len of the dataset',len(self.output_[:,0]))
         return len(self.output_[:,0])
     def __getitem__(self, idx):
         return self.input_[idx,:,:], self.output_[idx,:], self.init[idx,:], self.scaler[idx]

# Shape the inputs and outputs
input_seq = np.zeros(shape=(num_train*(frames-window),window,input_len))
output_seq = np.zeros(shape=(num_train*(frames-window),output_len))
input_test = np.zeros(shape=(num_test*(frames-window),window,input_len))
output_test = np.zeros(shape=(num_test*(frames-window),output_len))
ini_train = np.zeros(shape=(num_train*(frames-window),output_len))
ini_test = np.zeros(shape=(num_test*(frames-window),output_len))
# Setting up inputs and outputs
sample = 0
for run in range(num_train):
    lstm_snapshot = frac_train[run,:,:]
    for t in range(window,frames):
        input_seq[sample,:,:output_len] = lstm_snapshot[t-window:t,:]
        input_seq[sample,:,output_len:-1] = param_train[run,:]
        input_seq[sample,:,-1] = t/(frames-1) 
        output_seq[sample,:] = lstm_snapshot[t,:]
        ini_train[sample,:] = frac_train_ini[run,:]
        sample = sample + 1
#print(np.max(output_seq))
sample = 0
for run in range(num_test):
    lstm_snapshot = frac_test[run,:,:]
    for t in range(window,frames):
        input_test[sample,:,:output_len] = lstm_snapshot[t-window:t,:]
        input_test[sample,:,output_len:-1] = param_test[run,:]
        input_test[sample,:,-1] = t/(frames-1) 
        output_test[sample,:] = lstm_snapshot[t,:]
        ini_test[sample,:] = frac_test_ini[run,:]
        sample = sample + 1
        
input_dat = torch.from_numpy(input_seq)
input_test_pt = torch.from_numpy(input_test)
output_dat = torch.from_numpy(output_seq)
output_test_pt = torch.from_numpy(output_test)

scaler_train = np.tile(scaler_lstm[window:],num_train)
scaler_test = np.tile(scaler_lstm[window:],num_test)


train_loader = PrepareData(input_seq, output_seq, ini_train, scaler_train)
test_loader = PrepareData(input_test, output_test, ini_test, scaler_test)
#train_loader = DataLoader(train_loader, batch_size = num_train*(frames-window), shuffle=False)
train_loader = DataLoader(train_loader, batch_size = 64, shuffle=False)
test_loader = DataLoader(test_loader, batch_size = num_test*(frames-window), shuffle=False)

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
        self.lstm = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)
        self.project = nn.Linear(hidden_dim, output_len) # input = [batch, dim] 
        self.linear = nn.Linear(output_len,output_len)
        #self.frac_ini = frac_ini
    def forward(self, input_frac, frac_ini, scaler):
        
        lstm_out, _ = self.lstm(input_frac)  # output range [-1,1]
        target = self.project(lstm_out[:,-1,:]) # project to the desired shape
        #target = F.dropout(target, p=0.1)
       # frac = F.softmax(target,dim=1) # dim0 is the batch, dim1 is the vector
        target = F.relu(target+frac_ini)  # frac_ini here is necessary to keep 
        frac = F.normalize(target,p=1,dim=1)-frac_ini # dim0 is the batch, dim1 is the vector
        frac = scaler.view(-1,1)*frac
     #   print(scaler.shape,frac.shape)
      #  frac = scaler.unsqueeze(dim=1)*frac
        return frac

def scaled_loss(output, target, runs, pred, scaler):

    loss = torch.sum((scaler[:,np.newaxis]*(output-target))**2)/(runs*pred)

    return loss

def LSTM_train(model, num_epochs, train_loader, test_loader):
    
    #torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
                                 #weight_decay=1e-5) # <--
 #   optimizer = AdaBound(model.parameters(),lr=learning_rate,final_lr=0.1)
  #  outputs = []
    for epoch in range(num_epochs):
      #if epoch < 100:
      # optimizer = torch.optim.Adam(model.parameters(),
      #                               lr=learning_rate)
      if epoch==num_epochs/2: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
      for  ix, (I_train, O_train, ini_train, scaler_train) in enumerate(train_loader):   

         #print(I_train.shape)
         recon = model(I_train,ini_train,scaler_train)
         loss = criterion(recon, O_train)
        # print(recon,O_train)
         #loss = scaled_loss(recon, O_train, num_train, pred_frames, scaler_train)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
        # optimizer.zero_grad() 
      for  ix, (I_test, O_test, ini_test, scaler_test) in enumerate(test_loader):
        pred = model(I_test,ini_test,scaler_test)
        test_loss = criterion(pred, O_test)
        #test_loss = scaled_loss(pred, O_test, num_test, pred_frames, scaler_test)
        #print(recon.shape,O_train.shape,pred.shape, O_test.shape)
        print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(loss), float(test_loss)))
        # outputs.append((epoch, data, recon),)
        
    return model 


model = LSTM_soft(input_len, output_len, hidden_dim, LSTM_layer)
model = model.double()
if device=='cuda':
  model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)

#scaler_lstm = pred_frames*np.ones(pred_frames)/(np.arange(pred_frames)+pred_frames)
#scaler_lstm = (frames-np.arange(frames))*linear_fact
#scaler_train = todevice(np.tile(scaler_lstm[window:],num_train))
#scaler_test = todevice(np.tile(scaler_lstm[window:],num_test)) 

#print(scaler_train)


if model_exist:

  model.load_state_dict(torch.load('./lstmmodel'))
  model.eval()   
else: 
  start = time.time()
  model=LSTM_train(model, num_epochs, train_loader, test_loader)
  end = time.time()
  print('training time',-start+end)
  torch.save(model.state_dict(), './lstmmodel')



## plot to check if the construction is reasonable
evolve_runs = 10 #num_test
frac_out = np.zeros((evolve_runs,frames,G))
train_dat = np.zeros((evolve_runs,window,input_len))

frac_out_true = output_test_pt.detach().numpy()[:pred_frames,:]

# evole physics based on trained network
frac_out_info = frac_test[:evolve_runs,:window,:]
frac_out[:,:window,:] = frac_out_info
train_dat[:evolve_runs,:,output_len:-1] = param_test[:evolve_runs,np.newaxis,:]
for i in range(pred_frames):
    train_dat[:evolve_runs,:,:output_len] = frac_out_info
    train_dat[:evolve_runs,:,-1] = (i+window)/(frames-1) 
    #print(train_dat.shape)
   # train_dat = torch.from_numpy(np.vstack(frac_out_info,))
    frac_new_vec = tohost(model(todevice(train_dat), todevice(frac_test_ini[:evolve_runs,:]),todevice(scaler_lstm[[window+i]])) ) 
    #print('timestep ',i)
    #print('predict',frac_new_vec/scaler_lstm[window+i])
    #print('true',frac_out_true[i,:]/scaler_lstm[window+i])
    frac_out[:evolve_runs,window+i,:] = frac_new_vec
    #print(frac_new_vec)
    frac_out_info = np.concatenate((frac_out_info[:evolve_runs,1:,:],frac_new_vec[:evolve_runs,np.newaxis,:]),axis=1)
    
frac_out = frac_out/scaler_lstm[np.newaxis,:,np.newaxis] + frac_test_ini[:evolve_runs,np.newaxis,:]
#print(frac_out)
#frac_out_trained = output_test_pt.detach().numpy()[plot_idx*pred_frames:(plot_idx+1)*pred_frames,:]
#frac_out = np.vstack((frac_out_info, frac_out_trained))
#print(frac_out)
#print(frac_out_trained)

# ea

for plot_idx in range( evolve_runs ):  # in test dataset
   print('plot_id,run_id', plot_idx)
   print('seq', param_test[plot_idx,:])
   #frac_out_true = output_test_pt.detach().numpy()[plot_idx*pred_frames:(plot_idx+1)*pred_frames,:]
   frame_idx=plot_idx
   
   alpha_true = np.asarray(f['alpha'])[frame_idx*fnx*fny:(frame_idx+1)*fnx*fny]
   aseq_test = aseq_asse[(num_train+frame_idx)*G:(num_train+frame_idx+1)*G]
   tip_y = tip_y_asse[(num_train+frame_idx)*frames:(num_train+frame_idx+1)*frames]
   plot_real(x,y,alpha_true,plot_idx)
   plot_reconst(G,x,y,aseq_test,tip_y,alpha_true,frac_out[plot_idx,:,:].T,plot_idx)






