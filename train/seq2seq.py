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
from plot_funcs import plot_reconst,plot_real, plot_IO
from torch.utils.data import Dataset, DataLoader
import glob, os, re, sys, importlib
from check_data_quality import check_data_quality
#from melt_pool import *
from G_E import *
# global parameters
host='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device=host
print('device',device)
model_exist = False

param_list = ['anis','G0','Rmax']

print('train, test', num_train, num_test)
print('frames, window', frames, window)

# global information that apply for every run
#datasets = glob.glob('../../mulbatch_train/ML_PF10_train1000_test100_Mt47024_grains8_frames25_anis*_G05.000_Rmax1.000_seed*_rank0.h5')
#datasets = glob.glob('../../ML_mask_PF10_train500_test50_Mt70536_grains20_frames27_anis0.130_G05.000_Rmax1.000_seed2_rank0.h5')
#datasets = glob.glob('../../ML_PF10_train200_test20_Mt23274_grains8_frames25_anis0.050_G01.000_Rmax1.000_seed1_rank0.h5')
datasets = glob.glob(data_dir)
print('dataset list',datasets,' and size',len(datasets))
filename = datasets[0]
#filename = filebase+str(2)+ '_rank0.h5'
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


frac_all = np.zeros((num_runs,frames,G)) #run*frames*vec_len
param_all = np.zeros((num_runs,G+param_len))
y_all = np.zeros((num_runs,frames))

for batch_id in range(num_batch):
  fname =datasets[batch_id]; print(fname)
  f = h5py.File(str(fname), 'r') 
  aseq_asse = np.asarray(f['sequence'])
  frac_asse = np.asarray(f['fractions'])
  tip_y_asse = np.asarray(f['y_t'])
  number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
  print(number_list[6],number_list[7])
  # compile all the datasets interleave
  for run in range(batch):
    aseq = aseq_asse[run*G:(run+1)*G]  # 1 to 10
    tip_y = tip_y_asse[run*frames:(run+1)*frames]
#    Color = (aseq-3)/2        # normalize C to [-1,1]
    Color = (aseq-5.5)/4.5
    #print('angle sequence', Color)
    frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
    frac = frac.T
    frac_all[run*num_batch+batch_id,:,:] = frac
    y_all[run*num_batch+batch_id,:] = tip_y 
    param_all[run*num_batch+batch_id,:G] = Color
    param_all[run*num_batch+batch_id,G] = float(number_list[6])
    param_all[run*num_batch+batch_id,G+1] = float(number_list[7])/100 
#print(tip_y_asse[frames::frames])
# trained dataset need to be randomly selected:

weird_sim = check_data_quality(frac_all, param_all, y_all, G, frames)

### divide train and validation

idx =  np.arange(num_runs) # you can permute the order of train here
np.random.seed(seed)
#np.random.shuffle(idx[:-1])
#print(idx)
frac_train = frac_all[idx[:num_train],:,:]
frac_test = frac_all[idx[num_train_all:],:,:]
param_train = param_all[idx[:num_train],:]
param_test = param_all[idx[num_train_all:],:]

weird_sim = np.array(weird_sim)[np.array(weird_sim)<num_train]
frac_train = np.delete(frac_train,weird_sim,0)
num_train -= len(weird_sim) 
print('actual num_train',num_train)

## calculate the mask first


## subtract the initial part of the sequence, so we can focus on the change

frac_train_ini = frac_train[:,0,:]
frac_test_ini = frac_test[:,0,:]

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

     def __init__(self, input_, output_, init, scaler, mask):
          if not torch.is_tensor(input_):
              self.input_ = todevice(input_)
          if not torch.is_tensor(output_):
              self.output_ = todevice(output_)
          if not torch.is_tensor(init):
              self.init = todevice(init)
          if not torch.is_tensor(scaler):
              self.scaler = todevice(scaler)
          if not torch.is_tensor(mask):
              self.mask = todevice(mask)
     def __len__(self):
         #print('len of the dataset',len(self.output_[:,0]))
         return len(self.output_[:,0])
     def __getitem__(self, idx):
         return self.input_[idx,:,:], self.output_[idx,:], self.init[idx,:], self.scaler[idx], self.mask[idx,:]

# Shape the inputs and outputs
input_seq = np.zeros(shape=(num_train*sam_per_run, window, input_len))
output_seq = np.zeros(shape=(num_train*sam_per_run, out_win, output_len))
input_test = np.zeros(shape=(num_test*sam_per_run, window, input_len))
output_test = np.zeros(shape=(num_test*sam_per_run, out_win, output_len))

ini_train = np.zeros(shape=(num_train*sam_per_run, output_len))
ini_test = np.zeros(shape=(num_test*sam_per_run, output_len))

# Setting up inputs and outputs
# input t-window to t-1
# output t to t+win_out-1
sample = 0
for run in range(num_train):
    lstm_snapshot = frac_train[run,:,:]
    for t in range(window,frames-(out_win-1)):
        input_seq[sample,:,:output_len] = lstm_snapshot[t-window:t,:]
        input_seq[sample,:,output_len:-1] = param_train[run,:]
        input_seq[sample,:,-1] = t/(frames-1) 
        output_seq[sample,:,:] = lstm_snapshot[t:t+out_win,:]
        ini_train[sample,:] = frac_train_ini[run,:]
        sample = sample + 1
#print(np.max(output_seq))
sample = 0
for run in range(num_test):
    lstm_snapshot = frac_test[run,:,:]
    for t in range(window,frames-(out_win-1)):
        input_test[sample,:,:output_len] = lstm_snapshot[t-window:t,:]
        input_test[sample,:,output_len:-1] = param_test[run,:]
        input_test[sample,:,-1] = t/(frames-1) 
        output_test[sample,:,:] = lstm_snapshot[t:t+out_win,:]
        ini_test[sample,:] = frac_test_ini[run,:]
        sample = sample + 1
        


scaler_train = np.zeros((num_train*sam_per_run,out_win)) ## this is for the output scaling
scaler_test = np.zeros((num_test*sam_per_run,out_win))
for i in range(out_win):
    scaler_train[:,i] = np.tile(scaler_lstm[window+i:frames-(out_win-1)+i],num_train)  
    scaler_test[:,i] = np.tile(scaler_lstm[window+i:frames-(out_win-1)+i],num_test)

## scaler 

scaler_train_p = np.tile(scaler_lstm[window-1:frames-out_win],num_train)
scaler_test_p = np.tile(scaler_lstm[window-1:frames-out_win],num_test)

mask_train = (input_seq[:,-1,:output_len]/scaler_train_p[:,np.newaxis]+ini_train>1e-3)*np.ones((num_train*sam_per_run,output_len))
mask_test = (input_test[:,-1,:output_len]/scaler_test_p[:,np.newaxis]+ini_test>1e-3)*np.ones((num_test*sam_per_run,output_len))
#print(mask_train)
train_loader = PrepareData(input_seq, output_seq, ini_train, scaler_train, mask_train)
test_loader = PrepareData(input_test, output_test, ini_test, scaler_test, mask_test)

#train_loader = DataLoader(train_loader, batch_size = num_train*(frames-window), shuffle=False)
train_loader = DataLoader(train_loader, batch_size = 64, shuffle=True)
test_loader = DataLoader(test_loader, batch_size = num_test*(frames-window), shuffle=False)


class Decoder(nn.Module):
    def __init__(self,input_len,output_len,hidden_dim,num_layer):
        super(Decoder, self).__init__()
        self.input_len = input_len 
        self.output_len = output_len  
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lstm_decoder = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)    
        self.project = nn.Linear(hidden_dim, output_len)
    def forward(self,input_frac,hidden,cell,frac_ini,scaler,mask):
        output, (hidden, cell) = self.lstm_decoder(input_frac.unsqueeze(dim=1), (hidden,cell) )
        target = self.project(output[:,-1,:])   # project last layer output to the desired shape
        target = F.relu(target+frac_ini)         # frac_ini here is necessary to keep
        frac = F.normalize(target,p=1,dim=-1)-frac_ini   # normalize the fractions
        frac = scaler.unsqueeze(dim=-1)*frac     # scale the output based on the output frame
        
        return frac, hidden, cell
# The model
class LSTM_soft(nn.Module):
    def __init__(self,input_len,output_len,hidden_dim,num_layer,out_win,decoder):
        super(LSTM_soft, self).__init__()
        self.input_len = input_len
        self.output_len = output_len  
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.out_win = out_win
        self.lstm_encoder = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)
        self.decoder = decoder
        #self.project = nn.Linear(hidden_dim, output_len) # input = [batch, dim] 
     #   self.linear = nn.Linear(output_len,output_len)
      
    def forward(self, input_frac, frac_ini, scaler, mask):
        
        output_frac = torch.zeros(input_frac.shape[0],self.out_win,self.output_len)
        ## step 1 encode the input to hidden and cell state
        encode_out, (hidden, cell) = self.lstm_encoder(input_frac)  # output range [-1,1]
        ## step 2 start with "equal vector", the last 
        input_1seq = input_frac[:,-1,:]  ## the ancipated output frame is t
        ## step 3 for loop decode the time series one-by-one
        for i in range(self.out_win):
            output, hidden, cell = self.decoder(input_1seq, hidden, cell, frac_ini, scaler[:,i], mask)
            output_frac[:,i,:] = output
            input_1seq[:,:self.output_len] = output
            input_1seq[:,-1] += 1.0/(frames-1)  ## time tag 
            
        return output_frac



def LSTM_train(model, num_epochs, train_loader, test_loader):
    
    #torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
                                 #weight_decay=1e-5) # <--
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
 #   optimizer = AdaBound(model.parameters(),lr=learning_rate,final_lr=0.1)
  #  outputs = []
    for epoch in range(num_epochs):
      #if epoch < 100:
      # optimizer = torch.optim.Adam(model.parameters(),
      #                               lr=learning_rate)
      if epoch==num_epochs-10: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
      for  ix, (I_train, O_train, ini_train, scaler_train, mask_train) in enumerate(train_loader):   

         #print(I_train.shape)
         #recon = model(I_train,ini_train,scaler_train)
         loss = criterion(model(I_train,ini_train,scaler_train, mask_train), O_train)
        # print(recon,O_train)
         #loss = scaled_loss(recon, O_train, num_train, pred_frames, scaler_train)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
        # optimizer.zero_grad() 
      for  ix, (I_test, O_test, ini_test, scaler_test, mask_test) in enumerate(test_loader):
        #pred = model(I_test,ini_test,scaler_test)
        test_loss = criterion(model(I_test,ini_test,scaler_test, mask_test), O_test)
        #test_loss = scaled_loss(pred, O_test, num_test, pred_frames, scaler_test)
        #print(recon.shape,O_train.shape,pred.shape, O_test.shape)
      print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(loss), float(test_loss)))
        # outputs.append((epoch, data, recon),)
      train_list.append(float(loss))
      test_list.append(float(test_loss))       
      scheduler.step()
    return model 

decoder = Decoder(input_len,output_len,hidden_dim, LSTM_layer)
model = LSTM_soft(input_len, output_len, hidden_dim, LSTM_layer, out_win, decoder)
model = model.double()
if device=='cuda':
  model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)




if model_exist:

  model.load_state_dict(torch.load('./lstmmodel'))
  model.eval()   
else: 
  train_list=[]
  test_list=[]
  start = time.time()
  model=LSTM_train(model, num_epochs, train_loader, test_loader)
  end = time.time()
  print('training time',-start+end)
  torch.save(model.state_dict(), './lstmmodel')
  fig, ax = plt.subplots() 
  ax.semilogy(train_list)
  ax.semilogy(test_list)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['training loss','validation loss'])
  plt.title('training time:'+str( "%d"%int( (end-start)/60 ) )+'min')
  plt.savefig('mul_batch_loss.png')
  
## plot to check if the construction is reasonable
evolve_runs = num_batch #num_test
frac_out = np.zeros((evolve_runs,frames,G))
train_dat = np.zeros((evolve_runs,window,input_len))

#frac_out_true = output_test[:pred_frames,:]

# evole physics based on trained network
frac_out_info = frac_test[:evolve_runs,:window,:]
frac_out[:,:window,:] = frac_out_info
train_dat[:evolve_runs,:,output_len:-1] = param_test[:evolve_runs,np.newaxis,:]
alone = pred_frames%out_win
pack = pred_frames-alone
for i in range(0,pred_frames,out_win):
    train_dat[:evolve_runs,:,:output_len] = frac_out_info
    train_dat[:evolve_runs,:,-1] = (i+window)/(frames-1) ## the first output time
    mask_before = (train_dat[:,-1,:output_len]/scaler_lstm[[window+i-1]]+frac_test_ini[:evolve_runs,:]>1e-3)*np.ones((evolve_runs,output_len))
    frac_new_vec = tohost(model(todevice(train_dat), todevice(frac_test_ini[:evolve_runs,:]),todevice(scaler_lstm[window+i:window+i+out_win]), todevice(mask_before) ) ) 
    #print('timestep ',i)
    #print('predict',frac_new_vec/scaler_lstm[window+i])
    #print('true',frac_out_true[i,:]/scaler_lstm[window+i])
    if i>=pack:
        frac_out[:evolve_runs,-alone:,:] = frac_new_vec[:alone,:,:]
    else: frac_out[:evolve_runs,window+i:window+i+out_win,:] = frac_new_vec
    #print(frac_new_vec)
    frac_out_info = np.concatenate((frac_out_info[:evolve_runs,out_win:,:],frac_new_vec),axis=1)
    
frac_out = frac_out/scaler_lstm[np.newaxis,:,np.newaxis] + frac_test_ini[:evolve_runs,np.newaxis,:]


for batch_id in range(num_batch): 
 fname = datasets[batch_id] 
 f = h5py.File(fname, 'r')
 aseq_asse = np.asarray(f['sequence'])
 frac_asse = np.asarray(f['fractions'])
 tip_y_asse = np.asarray(f['y_t'])
 for plot_idx in range( int(evolve_runs/num_batch) ):  # in test dataset
   print('plot_id,batch_id', plot_idx, batch_id)
   data_id = plot_idx*num_batch+batch_id
   print('seq', param_test[data_id,:])
   #frac_out_true = output_test_pt.detach().numpy()[plot_idx*pred_frames:(plot_idx+1)*pred_frames,:]
   frame_idx=plot_idx  # here the idx means the local id of the test part (last 100)
   
   alpha_true = np.asarray(f['alpha'])[frame_idx*fnx*fny:(frame_idx+1)*fnx*fny]
   aseq_test = aseq_asse[(num_train_b+frame_idx)*G:(num_train_b+frame_idx+1)*G]
   tip_y = tip_y_asse[(num_train_b+frame_idx)*frames:(num_train_b+frame_idx+1)*frames]
   #plot_real(x,y,alpha_true,plot_idx)
   #plot_reconst(G,x,y,aseq_test,tip_y,alpha_true,frac_out[plot_idx,:,:].T,plot_idx)
   # get the parameters from dataset name
   G0 = param_test[data_id,G+1]
   Rmax = 1 
   anis = param_test[data_id,G]
   plot_IO(anis,G0,Rmax,G,x,y,aseq_test,tip_y,alpha_true,frac_out[plot_idx*num_batch+batch_id,:,:].T,window,data_id)




