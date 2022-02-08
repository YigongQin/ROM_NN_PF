#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:41:18 2021

@author: yigongqin
"""
from math import pi
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
from plot_funcs import plot_IO, miss_rate
from torch.utils.data import Dataset, DataLoader
import glob, os, re, sys, importlib
from check_data_quality import check_data_quality
from models import *
import matplotlib.tri as tri
from split_merge_reini import split_grain, merge_grain
from scipy.interpolate import griddata
torch.cuda.empty_cache()

mode = sys.argv[1]
if mode == 'train': from G_E_R import *
elif mode == 'test': from G_E_test import *
elif mode == 'ini': from G_E_ini import *
else: raise ValueError('mode not specified')
print('the mode is', mode)
# global parameters
host='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device=host
print('device',device)
model_exist = False
if mode == 'test': model_exist = True
noPDE = True
param_list = ['anis','G0','Rmax']

print('(input data) train, test', num_train, num_test)

datasets = sorted(glob.glob(data_dir))
print('dataset dir',data_dir,' and size',len(datasets))
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

def get_data(num_runs, num_batch, datasets):
  frac_all = np.zeros((num_runs,frames,G)) #run*frames*vec_len
  param_all = np.zeros((num_runs,param_len))
  y_all = np.zeros((num_runs,frames))
  area_all = np.zeros((num_runs,frames,G))
  G_list = []
  R_list = []
  e_list = []


  for batch_id in range(num_batch):
    fname =datasets[batch_id]; #print(fname)
    f = h5py.File(str(fname), 'r') 
    #aseq_asse = np.asarray(f['sequence'])
    aseq_asse = np.asarray(f['angles'])
    frac_asse = np.asarray(f['fractions'])
    area_asse = np.asarray(f['extra_area'])
    tip_y_asse = np.asarray(f['y_t'])
    number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
    #print(number_list[6],number_list[7],number_list[8])
    e_list.append(number_list[6])
    G_list.append(number_list[7])
    R_list.append(number_list[8])
    
    # compile all the datasets interleave
    for run in range(batch):
      aseq = aseq_asse[run*(G+1):(run+1)*(G+1)]  # 1 to 10
      tip_y = tip_y_asse[run*frames:(run+1)*frames]
  #    Color = (aseq-3)/2        # normalize C to [-1,1]
      #Color = (aseq-5.5)/4.5
      Color = - ( 2*(aseq[1:] + pi/2)/(pi/2) - 1 )
      #print('angle sequence', Color)
      frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((frames,G))  # grains coalese, include frames
      area = (area_asse[run*G*frames:(run+1)*G*frames]).reshape((frames,G))  # grains coalese, include frames
      #if run<1: print(frac) 
      frac_all[run*num_batch+batch_id,:,:] = frac
      y_all[run*num_batch+batch_id,:] = tip_y 
      area_all[run*num_batch+batch_id,:,:] = area
      param_all[run*num_batch+batch_id,:G] = Color
      param_all[run*num_batch+batch_id,G] = 2*float(number_list[6])
      param_all[run*num_batch+batch_id,G+1] = 1 - np.log10(float(number_list[7]))/np.log10(100) 
      param_all[run*num_batch+batch_id,G+2] = float(number_list[8])

  return frac_all, param_all, y_all, area_all, G_list, R_list, e_list

frac_train, param_train, y_train, area_train, G_list, R_list, e_list = get_data(num_train, num_train, datasets)
testsets = sorted(glob.glob(valid_dir))
frac_test, param_test, y_test, area_test, _ , _ , _= get_data(num_test, num_test, testsets)
#print(tip_y_asse[frames::frames])
# trained dataset need to be randomly selected:

if skip_check == False:
 weird_sim = check_data_quality(frac_train, param_train, y_train, G, frames)
else: weird_sim=[]
### divide train and validation

idx =  np.arange(num_runs) # you can permute the order of train here
np.random.seed(seed)
#np.random.shuffle(idx[:-1])

print('nan', np.where(np.isnan(frac_train)))
weird_sim = np.array(weird_sim)[np.array(weird_sim)<num_train]
print('throw away simulations',weird_sim)
#### delete the data in the actual training fractions and parameters
frac_train = np.delete(frac_train,weird_sim,0)
param_train = np.delete(param_train,weird_sim,0)
#idx_all = np.concatenate((np.delete(idx[:num_train],weird_sim,0),idx[num_train_all:])) 
num_train -= len(weird_sim) 

assert num_train==frac_train.shape[0]==param_train.shape[0]
assert num_test==frac_test.shape[0]==param_test.shape[0]
assert param_train.shape[1]==param_len
num_all = num_train + num_test

print('actual num_train',num_train)
print('total frames',frames,'in_win',window,'out_win',out_win)
print('epoch', num_epochs, 'learning rate',learning_rate)
print('1d grid size (number of grains)', G)
print('param length', param_len)
print('========== architecture ========')
print('type -- LSTM')
print('hidden dim', hidden_dim, 'number of layers', LSTM_layer)
print('convolution kernel size', kernel_size)
print('========== architecture ========')

def todevice(data):
    return torch.from_numpy(data).to(device)
def tohost(data):
    return data.detach().to(host).numpy()

class PrepareData(Dataset):

     def __init__(self, input_, output_, param, area):
          if not torch.is_tensor(input_):
              self.input_ = todevice(input_)
          if not torch.is_tensor(output_):
              self.output_ = todevice(output_)
          if not torch.is_tensor(param):
              self.param = todevice(param)
          if not torch.is_tensor(area):
              self.area = todevice(area)
     def __len__(self):
         #print('len of the dataset',len(self.output_[:,0]))
         return len(self.output_[:,0])
     def __getitem__(self, idx):
         return self.input_[idx,:,:], self.output_[idx,:,:], self.param[idx,:], self.area[idx,:]
     
# stack information
frac_all = np.concatenate( (frac_train, frac_test), axis=0)
param_all = np.concatenate( (param_train, param_test), axis=0)


y_all  = np.concatenate( (y_train, y_test), axis=0)
dy_all  = np.diff(y_all, axis=1) 
if mode == 'ini':
    dy_all = np.concatenate((dy_all[:,[0]]*0,dy_all),axis=-1)  ##extrapolate dy at t=0
else: dy_all = np.concatenate((dy_all[:,[0]],dy_all),axis=-1)
dy_all = dy_all/y_norm

## add area 

area_all  = np.concatenate( (area_train, area_test), axis=0)
darea_all = area_all/area_norm   ## frac norm is fixed in the code
#darea_all = np.concatenate((darea_all[:,[0],:],darea_all),axis=1) ##extrapolate dfrac at t=0
area_coeff = y_norm*fnx/dx/area_norm
## subtract the initial part of the sequence, so we can focus on the change

frac_ini = frac_all[:,0,:]

dfrac_all = np.diff(frac_all, axis=1)/frac_norm   ## frac norm is fixed in the code
if mode == 'ini':
    dfrac_all = np.concatenate((dfrac_all[:,[0],:]*0,dfrac_all),axis=1) ##extrapolate dfrac at t=0
else: dfrac_all = np.concatenate((dfrac_all[:,[0],:],dfrac_all),axis=1) 
## scale the frac according to the time frame 

#frac_all *= scaler_lstm[np.newaxis,:,np.newaxis]

seq_all = np.concatenate( ( frac_all, dfrac_all, darea_all, dy_all[:,:,np.newaxis] ), axis=-1) 
param_all = np.concatenate( (frac_ini, param_all), axis=1)
param_len = param_all.shape[1]
assert frac_all.shape[0] == param_all.shape[0] == y_all.shape[0] == num_all
assert param_all.shape[1] == (2*G+4)
assert seq_all.shape[2] == (3*G+1)


# Shape the inputs and outputs
trunc = 0
print('truncate train len', trunc)
sam_per_run-=trunc
num_all_traj = int(1*num_train)
all_samp = num_all*sam_per_run + num_all_traj*trunc

input_seq = np.zeros((all_samp, window, 3*G+1))
input_param = np.zeros((all_samp, param_len))
output_seq = np.zeros((all_samp, out_win, 2*G+1))
output_area = np.zeros((all_samp, G))
assert input_param.shape[1]==param_all.shape[1]

###### input_seq last dim seq: frac, y #######
###### input_param last dim seq: ini, phase_field, ek, G, time #######


train_sam=num_train*sam_per_run + num_all_traj*trunc
test_sam=num_test*sam_per_run
print('train samples', train_sam)

# Setting up inputs and outputs
# input t-window to t-1
# output t to t+win_out-1
sample = 0
for run in range(num_all):
    lstm_snapshot = seq_all[run,:,:]
    if run < num_all_traj:
        end_frame = sam_per_run+window+trunc
    else: end_frame = sam_per_run+window
        
    for t in range(window, end_frame):
        
        input_seq[sample,:,:] = lstm_snapshot[t-window:t,:]        
        output_seq[sample,:,:] = lstm_snapshot[t:t+out_win,G:]
        #output_seq[sample,:,:] = np.concatenate((lstm_snapshot[t:t+out_win,G:2*G],lstm_snapshot[t:t+out_win,-1:]),axis=-1)        
        input_param[sample,:-1] = param_all[run,:-1]  # except the last one, other parameters are independent on time
        input_param[sample,-1] = t*dt 
        output_area[sample,:] = np.sum(area_all[run,t-1:t+out_win-1,:],axis=0)
        
        sample = sample + 1

assert sample==input_seq.shape[0]==train_sam+test_sam
assert np.all(np.absolute(input_param[:,G:])>1e-6)

#sio.savemat('input_trunc.mat',{'input_seq':input_seq,'input_param':input_param})
torch.manual_seed(35)


def con_samlpe(a, b):
    
    return np.concatenate((a,b), axis=0)

def augmentation(input_seq, output_seq, input_param, output_area):
    
    input_seq_re   = np.concatenate(( np.flip( input_seq [:,:,:-1],-1), input_seq[:,:,-1:]  ), axis = -1)
    output_seq_re  = np.concatenate(( np.flip( output_seq[:,:,:-1],-1), output_seq[:,:,-1:]  ), axis = -1)
    input_param_re = np.concatenate(( np.flip(input_param[:,:G], -1), np.flip(input_param[:,G:2*G], -1), input_param[:,2*G:]), axis = -1)   
    output_area_re = np.flip( output_area[:,:],-1)
    
    return con_samlpe(input_seq, input_seq_re), con_samlpe(output_seq, output_seq_re),\
           con_samlpe(input_param, input_param_re), con_samlpe(output_area, output_area_re)


#data_para = augmentation( input_seq[:train_sam,:,:], output_seq[:train_sam,:,:], \
#                                           input_param[:train_sam,:], output_area[:train_sam,:] )
data_para = [input_seq[:train_sam,:,:], output_seq[:train_sam,:,:], \
                                           input_param[:train_sam,:], output_area[:train_sam,:]] 

if not mode=='test':
   train_loader = PrepareData( data_para[0], data_para[1], data_para[2], data_para[3] )
   train_loader = DataLoader(train_loader, batch_size = 64, shuffle=True)

test_loader  = PrepareData(input_seq[train_sam:,:,:], output_seq[train_sam:,:,:], input_param[train_sam:,:], output_area[train_sam:,:])

test_loader = DataLoader(test_loader, batch_size = test_sam//8, shuffle=False)

def train(model, num_epochs, train_loader, test_loader):
    
    #torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
                                 #weight_decay=1e-5) # <--
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5, last_epoch=-1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
 #   optimizer = AdaBound(model.parameters(),lr=learning_rate,final_lr=0.1)

    train_loss = 0
    count = 0
    for  ix, (I_train, O_train, P_train, A_train) in enumerate(train_loader):   
        count += I_train.shape[0]
        recon, area_train = model(I_train, P_train)
        train_loss += I_train.shape[0]*float(criterion(recon, O_train)) #+ 0.01*out_win/dt*criterion(area_train, A_train)
    train_loss/=count

    test_loss = 0
    count = 0
    for  ix, (I_test, O_test, P_test, A_test) in enumerate(test_loader):      
        count += I_test.shape[0]
        pred, area_test = model(I_test, P_test)
        test_loss += I_test.shape[0]*float(criterion(pred, O_test)) #+ 0.01*out_win/dt*criterion(area_test, A_test)
    test_loss/=count

    print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(0, float(train_loss), float(test_loss)))
    train_list.append(float(train_loss))
    test_list.append(float(test_loss))  

    for epoch in range(num_epochs):
      #if epoch < 100:
      # optimizer = torch.optim.Adam(model.parameters(),
      #                               lr=learning_rate)
      if mode=='train' and epoch==num_epochs-10: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
      train_loss = 0
      count = 0
      for  ix, (I_train, O_train, P_train, A_train) in enumerate(train_loader):   
         count += I_train.shape[0]
         #print(I_train.shape[0])
         recon, area_train = model(I_train, P_train)
        # loss = criterion(model(I_train, P_train), O_train)
         loss = criterion(recon, O_train) #+ 0.01*out_win/dt*criterion(area_train, A_train)

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
         train_loss += I_train.shape[0]*float(loss)
        # exit() 
      train_loss/=count
      test_loss = 0
      count = 0
      for  ix, (I_test, O_test, P_test, A_test) in enumerate(test_loader):
        #print(I_test.shape[0])
        count += I_test.shape[0]
        pred, area_test = model(I_test, P_test)
        #test_loss = criterion(model(I_test, P_test), O_test)
        #print(criterion(pred, O_test) , out_win/dt*criterion(area_test, A_test))
        test_loss += I_test.shape[0]*float(criterion(pred, O_test)) #+ 0.01*out_win/dt*criterion(area_test, A_test)
 
      test_loss/=count
      print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(train_loss), float(test_loss)))
 
      train_list.append(float(loss))
      test_list.append(float(test_loss))       
      scheduler.step()

    return model 

#decoder = Decoder(input_len,output_len,hidden_dim, LSTM_layer)
#model = LSTM(input_len, output_len, hidden_dim, LSTM_layer, out_win, decoder, device)
#model = ConvLSTM_1step(3+param_len, hidden_dim, LSTM_layer, G, out_win, kernel_size, True, device)
if mode=='train' or mode == 'test': model = ConvLSTM_seq(10, hidden_dim, LSTM_layer, G_small, out_win, kernel_size, True, device, dt)
if mode=='ini': model = ConvLSTM_start(10, hidden_dim, LSTM_layer, G_small, out_win, kernel_size, True, device, dt)

model = model.double()
if device=='cuda':
  model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)




if model_exist:
  if mode == 'train' or mode== 'test':
    model.load_state_dict(torch.load('./lstmmodel'+sys.argv[2]))
    model.eval()  
  if mode == 'ini':  
    model.load_state_dict(torch.load('./ini_lstmmodel'+sys.argv[2]))
    model.eval() 
else: 
  train_list=[]
  test_list=[]
  start = time.time()
  model=train(model, num_epochs, train_loader, test_loader)
  end = time.time()
  print('training time',-start+end)
  if mode == 'train': torch.save(model.state_dict(), './lstmmodel'+sys.argv[2])
  if mode == 'ini': torch.save(model.state_dict(), './ini_lstmmodel'+sys.argv[2])
  fig, ax = plt.subplots() 
  ax.semilogy(train_list)
  ax.semilogy(test_list)
  txt = 'final train loss '+str('%1.2e'%train_list[-1])+' validation loss '+ str('%1.2e'%test_list[-1]) 
  fig.text(.5, .2, txt, ha='center')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['training loss','validation loss'])
  plt.title('training time:'+str( "%d"%int( (end-start)/60 ) )+'min')
  plt.savefig('mul_batch_loss.png')
  
## plot to check if the construction is reasonable
evolve_runs = num_test #num_test
frac_out = np.zeros((evolve_runs,frames,G)) ## final output
dfrac_out = np.zeros((evolve_runs,frames,G)) ## final output
dy_out = np.zeros((evolve_runs,frames))
darea_out = np.zeros((evolve_runs,frames,G))
left_grains = np.zeros((evolve_runs,frames,G))

seq_out = np.concatenate( ( frac_out, dfrac_out, darea_out, dy_out[:,:,np.newaxis] ), axis=-1) 

alone = pred_frames%out_win
pack = pred_frames-alone

param_test = param_all[num_train:,:]
param_dat = param_test[:evolve_runs,:]

seq_test = seq_all[num_train:,:,:]

frac_out[:,0,:] = seq_test[:,0,:G]
dy_out[:,0] = seq_test[:,0,-1]
darea_out[:,0,:] = seq_test[:,0,2*G:3*G]
left_grains[:,0,:] = np.cumsum(frac_out[:,0,:], axis=-1) - frac_out[:,0,:]

if noPDE == False:
    seq_dat = seq_test[:evolve_runs,:window,:]

    frac_out[:,:window,:] = seq_dat[:,:,:G]
    dy_out[:,:window] = seq_dat[:,:,-1]
    darea_out[:,:window,:] = seq_dat[:,:,2*G:3*G]

    param_dat, seq_dat, expand, left_coors = split_grain(param_dat, seq_dat, G_small, G)
else: 
    ini_model = ConvLSTM_start(10, hidden_dim, LSTM_layer_ini, G_small, window-1, kernel_size, True, device, dt)
    ini_model = ini_model.double()
    if device=='cuda':
       ini_model.cuda()
    init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
    print('total number of trained parameters for initialize model', init_total_params)
    ini_model.load_state_dict(torch.load('./ini_lstmmodel'+sys.argv[2]))
    ini_model.eval()

    seq_1 = seq_test[:,[0],:]   ## this can be generated randomly
    seq_1[:,:,-1]=0
    seq_1[:,:,G:2*G]=0
    print('sample', seq_1[0,0,:])

    param_dat_s, seq_1_s, expand, domain_factor, left_coors = split_grain(param_dat, seq_1, G_small, G)

    param_dat_s[:,-1] = dt
    domain_factor = size_scale*domain_factor
    seq_1_s[:,:,2*G_small:3*G_small] /= size_scale

    output_model = ini_model(todevice(seq_1_s), todevice(param_dat_s), todevice(domain_factor) )
    dfrac_new = tohost( output_model[0] ) 
    frac_new = tohost(output_model[1])

    dfrac_new[:,:,G_small:2*G_small] *= size_scale

    frac_out[:,1:window,:], dy_out[:,1:window], darea_out[:,1:window,:], left_grains[:,1:window,:] \
        = merge_grain(frac_new, dfrac_new[:,:,-1], dfrac_new[:,:,G_small:2*G_small], G_small, G, expand, domain_factor, left_coors)

    seq_dat_s = np.concatenate((seq_1_s,np.concatenate((frac_new, dfrac_new), axis = -1)),axis=1)
    if mode != 'ini':
      seq_dat_s[:,0,-1] = seq_dat_s[:,1,-1]
      seq_dat_s[:,0,G:2*G] = seq_dat_s[:,1,G:2*G] 
    #print(frac_new_vec.shape)

## write initial windowed data to out arrays

#print('the sub simulations', expand)

for i in range(0,pred_frames,out_win):
    
    param_dat[:,-1] = (i+window)*dt ## the first output time
    print('nondim time', (i+window)*dt)

    ## you may resplit the grains here

   # param_dat, seq_dat, expand, domain_factor, left_coors = split_grain(param_dat, seq_dat, G_small, G)

   # domain_factor = size_scale*domain_factor
    seq_dat_s[:,:,2*G_small:3*G_small] /= size_scale

    output_model = model(todevice(seq_dat_s), todevice(param_dat_s), todevice(domain_factor)  )
    dfrac_new = tohost( output_model[0] ) 
    frac_new = tohost(output_model[1])

    dfrac_new[:,:,G_small:2*G_small] *= size_scale


    #if i>=pack:
     #   frac_out[:,-alone:,:], dy_out[:,-alone:], darea_out[:,-alone:,:], left_grains[:,-alone:,:] \
    #    = merge_grain(frac_new[:,:alone,:], dfrac_new[:,:alone,-1], dfrac_new[:,:alone,G_small:2*G_small], G_small, G, expand, domain_factor, left_coors)
   # else: 

    frac_out[:,window+i:window+i+out_win,:], dy_out[:,window+i:window+i+out_win], darea_out[:,window+i:window+i+out_win,:], left_grains[:,window+i:window+i+out_win,:] \
    = merge_grain(frac_new, dfrac_new[:,:,-1], dfrac_new[:,:,G_small:2*G_small], G_small, G, expand, domain_factor, left_coors)
    
    seq_dat_s = np.concatenate((seq_dat_s[:,out_win:,:], np.concatenate((frac_new, dfrac_new), axis = -1) ),axis=1)

#frac_out = frac_out/(np.sum(frac_out, axis=-1)[:,:,np.newaxis])    
#frac_out = frac_out/scaler_lstm[np.newaxis,:,np.newaxis] + frac_test[:evolve_runs,[0],:]
dy_out = dy_out*y_norm
dy_out[:,0] = 0
y_out = np.cumsum(dy_out,axis=-1)+y_all[num_train:num_train+evolve_runs,[0]]

area_out = darea_out*area_norm
#darea_out[:,0,:] = 0
#area_out = np.cumsum(darea_out,axis=1)+area_all[num_train:num_train+evolve_runs,[0],:]
#print((y_out[0,:]))


miss_rate_param = np.zeros(num_test)
run_per_param = int(evolve_runs/num_batch)
if run_per_param <1: run_per_param = 1

if mode == 'test': valid_train = True
else: valid = False
valid_train = True
if valid_train:
  for batch_id in range(num_test): 
   fname = testsets[batch_id] 
   f = h5py.File(fname, 'r')
   aseq_asse = np.asarray(f['sequence'])
   angles_asse = np.asarray(f['angles'])
   frac_asse = np.asarray(f['fractions'])
   tip_y_asse = np.asarray(f['y_t'])
   area_asse = np.asarray(f['extra_area'])
   sum_miss = 0
   for plot_idx in range( run_per_param ):  # in test dataset

     data_id = plot_idx*num_batch+batch_id
     #print('seq', param_test[data_id,:])
     #frac_out_true = output_test_pt.detach().numpy()[plot_idx*pred_frames:(plot_idx+1)*pred_frames,:]
     frame_idx=plot_idx  # here the idx means the local id of the test part (last 100)
     
     alpha_true = np.asarray(f['alpha'])[frame_idx*fnx*fny:(frame_idx+1)*fnx*fny]
     aseq_test = aseq_asse[(num_train_b+frame_idx)*G:(num_train_b+frame_idx+1)*G]
     pf_angles = angles_asse[(num_train_b+frame_idx)*(G+1):(num_train_b+frame_idx+1)*(G+1)]
     pf_angles[1:] = pf_angles[1:]*180/pi + 90
     tip_y = tip_y_asse[(num_train_b+frame_idx)*frames:(num_train_b+frame_idx+1)*frames]
     extra_area = (area_asse[(num_train_b+frame_idx)*G*frames:(num_train_b+frame_idx+1)*G*frames]).reshape((frames,G))[train_frames-1,:]
     #print((tip_y))
     #plot_real(x,y,alpha_true,plot_idx)
     #plot_reconst(G,x,y,aseq_test,tip_y,alpha_true,frac_out[plot_idx,:,:].T,plot_idx)
     # get the parameters from dataset name
     G0 = np.float(G_list[batch_id])  #param_test[data_id,2*G+1]
     Rmax = np.float(R_list[batch_id]) 
     anis = np.float(e_list[batch_id])   #param_test[data_id,2*G]
     #plot_IO(anis,G0,Rmax,G,x,y,aseq_test,y_out[data_id,:],alpha_true,frac_out[data_id,:,:].T,window,data_id)
     #plot_IO(anis,G0,Rmax,G,x,y,aseq_test,y_out[data_id,:],alpha_true,frac_out[data_id,:,:].T,data_id, tip_y[train_frames-1],train_frames, pf_angles, extra_area, area_out[data_id,train_frames-1,:], left_grains[data_id,:,:].T)
     miss = miss_rate(anis,G0,Rmax,G,x,y,aseq_test,y_out[data_id,:],alpha_true,frac_out[data_id,:,:].T,data_id,tip_y[train_frames-1],train_frames, pf_angles, extra_area, area_out[data_id,train_frames-1,:], left_grains[data_id,:,:].T)
     sum_miss = sum_miss + miss
     print('plot_id,batch_id', plot_idx, batch_id,'miss%',miss)
   miss_rate_param[batch_id] = sum_miss/run_per_param


#fig, ax = plt.subplots() 

x = np.array(e_list,dtype=float)
y = np.array(G_list,dtype=float)
z = np.array(R_list,dtype=float)
u = np.array(miss_rate_param,dtype=float)

print(x)
print(y)
print(z)
print(u)
print('for model ', int(sys.argv[2]), 'the mean error', np.mean(u))

sio.savemat('2D_train'+str(num_train)+'_test'+str(num_test)+'_mode_'+mode+'.mat',{'frac_out':frac_out,'y_out':y_out,'e':x,'G':y,'R':z,'err':u,\
  'seq_all':seq_all,'param_all':param_all})

'''
xi = np.linspace(np.min(x), np.max(x), 1000)
yi = np.linspace(np.min(y), np.max(y), 1000)
X,Y= np.meshgrid(xi,yi)
Z = griddata((x, y), z, (X, Y),method='linear')


cntr = ax.contourf(X, Y, Z, vmin = 0.04, vmax =0.12,levels=np.linspace(0.04,0.12,1000), cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(x, y, 'ko', ms=8)
print(x, y, z)
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.xlabel(r'$\epsilon_k$')
plt.ylabel(r'$G\ (K/ \mu m)$')
plt.title('misclassification rate')
plt.savefig('miss_rate_trunc'+str(trunc)+mode+'.png',dpi=600)
'''

