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
from plot_funcs import plot_reconst, plot_real, plot_IO, miss_rate
from torch.utils.data import Dataset, DataLoader
import glob, os, re, sys, importlib
from check_data_quality import check_data_quality
from models import *
import matplotlib.tri as tri

mode = sys.argv[1]
if mode == 'train': from G_E import *
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
noPDE = False
param_list = ['anis','G0','Rmax']

print('(input data) train, test', num_train, num_test)

datasets = sorted(glob.glob(data_dir))
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
param_all = np.zeros((num_runs,param_len))
y_all = np.zeros((num_runs,frames))
G_list = []
R_list = []
e_list = []


for batch_id in range(num_batch):
  fname =datasets[batch_id]; print(fname)
  f = h5py.File(str(fname), 'r') 
  #aseq_asse = np.asarray(f['sequence'])
  aseq_asse = np.asarray(f['angles'])
  frac_asse = np.asarray(f['fractions'])
  tip_y_asse = np.asarray(f['y_t'])
  number_list=re.findall(r"[-+]?\d*\.\d+|\d+", datasets[batch_id])
  print(number_list[6],number_list[7],number_list[8])
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
    frac = (frac_asse[run*G*frames:(run+1)*G*frames]).reshape((G,frames), order='F')  # grains coalese, include frames
    frac = frac.T
    #if run<1: print(frac) 
    frac_all[run*num_batch+batch_id,:,:] = frac
    y_all[run*num_batch+batch_id,:] = tip_y 
    param_all[run*num_batch+batch_id,:G] = Color
    param_all[run*num_batch+batch_id,G] = 2*float(number_list[6])
    param_all[run*num_batch+batch_id,G+1] = 1 - np.log10(float(number_list[7]))/np.log10(100) 
#print(tip_y_asse[frames::frames])
# trained dataset need to be randomly selected:
sio.savemat('ini_data.mat',{'frac':frac_all[:,0,:],'y':y_all,'param':param_all})
if skip_check == False:
 weird_sim = check_data_quality(frac_all, param_all, y_all, G, frames)
else: weird_sim=[]
### divide train and validation

idx =  np.arange(num_runs) # you can permute the order of train here
np.random.seed(seed)
#np.random.shuffle(idx[:-1])
#print(idx)

## select num_train from num_train_all to frac_train, param_train
frac_train = frac_all[idx[:num_train],:,:]
frac_test = frac_all[idx[num_train_all:],:,:]
param_train = param_all[idx[:num_train],:]
param_test = param_all[idx[num_train_all:],:]
print('nan', np.where(np.isnan(frac_all)))
weird_sim = np.array(weird_sim)[np.array(weird_sim)<num_train]
print('throw away simulations',weird_sim)
#### delete the data in the actual training fractions and parameters
frac_train = np.delete(frac_train,weird_sim,0)
param_train = np.delete(param_train,weird_sim,0)
idx_all = np.concatenate((np.delete(idx[:num_train],weird_sim,0),idx[num_train_all:])) 
num_train -= len(weird_sim) 
assert num_train==frac_train.shape[0]==param_train.shape[0]
assert num_test==frac_test.shape[0]==param_test.shape[0]
assert param_all.shape[1]==param_len
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
y_norm = 0.5
y_all  = y_all[idx_all,:]
dy_all  = np.diff(y_all, axis=1) ## from here y_all means
dy_all = np.concatenate((dy_all[:,[0]],dy_all),axis=-1)
dy_all = dy_all/y_norm
## add area 
area_all = 0.5*dy_all[:,:-1,np.newaxis]*( frac_all[:,:-1,:] + frac_all[:,1:,:] )
assert area_all.shape[1]==frames-1

## subtract the initial part of the sequence, so we can focus on the change

frac_ini = frac_all[:,0,:]
frac_all = frac_all - frac_ini[:,np.newaxis,:]

## scale the frac according to the time frame 

scaler_lstm = scale(np.arange(frames)*dt,dt) # input to scale always 0 to 1
frac_all *= scaler_lstm[np.newaxis,:,np.newaxis]

seq_all = np.concatenate( ( frac_all[:,:,:], dy_all[:,:,np.newaxis] ), axis=-1) 
param_all = np.concatenate( (frac_ini, param_all), axis=1)
param_len = param_all.shape[1]
assert frac_all.shape[0] == param_all.shape[0] == y_all.shape[0] == num_all
assert param_all.shape[1] == (2*G+3)



# Shape the inputs and outputs
trunc = int(sys.argv[2])
print('truncate train len', trunc)
sam_per_run-=trunc
num_all_traj = int(0.1*num_train)
all_samp = num_all*sam_per_run + num_all_traj*trunc

input_seq = np.zeros((all_samp, window, G+1))
input_param = np.zeros((all_samp, param_len))
output_seq = np.zeros((all_samp, out_win, G+1))
output_area = np.zeros((all_samp, G))
assert input_param.shape[1]==param_all.shape[1]

###### input_seq last dim seq: frac, y #######
###### input_param last dim seq: ini, phase_field, ek, G, time #######


train_sam=num_train*sam_per_run + num_all_traj*trunc
test_sam=num_test*sam_per_run
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
        output_seq[sample,:,:] = lstm_snapshot[t:t+out_win,:]
        
        input_param[sample,:-1] = param_all[run,:-1]  # except the last one, other parameters are independent on time
        input_param[sample,-1] = t*dt 
        output_area[sample,:] = np.sum(area_all[run,t-1:t+out_win-1,:],axis=0)
        
        sample = sample + 1

assert sample==input_seq.shape[0]==train_sam+test_sam
assert np.all(np.absolute(input_param[:,G:])>1e-6)


torch.manual_seed(35)

if not mode=='test':
   train_loader = PrepareData(input_seq[:train_sam,:,:], output_seq[:train_sam,:,:], input_param[:train_sam,:], output_area[:train_sam,:])
   train_loader = DataLoader(train_loader, batch_size = 64, shuffle=True)

test_loader  = PrepareData(input_seq[train_sam:,:,:], output_seq[train_sam:,:,:], input_param[train_sam:,:], output_area[train_sam:,:])

test_loader = DataLoader(test_loader, batch_size = test_sam, shuffle=False)

def train(model, num_epochs, train_loader, test_loader):
    
    #torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
                                 #weight_decay=1e-5) # <--
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
 #   optimizer = AdaBound(model.parameters(),lr=learning_rate,final_lr=0.1)
  #  outputs = []
    for  ix, (I_test, O_test, P_test, A_test) in enumerate(test_loader):

        pred, area_test = model(I_test, P_test)
        #test_loss = criterion(model(I_test, P_test), O_test)
        #print(criterion(pred, O_test) , out_win/dt*criterion(area_test, A_test))
        test_loss = criterion(pred, O_test) #+ 0.01*out_win/dt*criterion(area_test, A_test)
        #test_loss = scaled_loss(pred, O_test, num_test, pred_frames, scaler_test)
        #print(recon.shape,O_train.shape,pred.shape, O_test.shape)
    print('Epoch:{}, valid loss:{:.6f}'.format(0, float(test_loss)))
    for epoch in range(num_epochs):
      #if epoch < 100:
      # optimizer = torch.optim.Adam(model.parameters(),
      #                               lr=learning_rate)
      if mode=='train' and epoch==num_epochs-10: optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
      for  ix, (I_train, O_train, P_train, A_train) in enumerate(train_loader):   

         #print(I_train.shape)
         recon, area_train = model(I_train, P_train)
        # loss = criterion(model(I_train, P_train), O_train)
         loss = criterion(recon, O_train) #+ 0.01*out_win/dt*criterion(area_train, A_train)
         #loss = scaled_loss(recon, O_train, num_train, pred_frames, scaler_train)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
       
      for  ix, (I_test, O_test, P_test, A_test) in enumerate(test_loader):
          
        pred, area_test = model(I_test, P_test)
        #test_loss = criterion(model(I_test, P_test), O_test)
        #print(criterion(pred, O_test) , out_win/dt*criterion(area_test, A_test))
        test_loss = criterion(pred, O_test) #+ 0.01*out_win/dt*criterion(area_test, A_test)
        #test_loss = scaled_loss(pred, O_test, num_test, pred_frames, scaler_test)
        #print(recon.shape,O_train.shape,pred.shape, O_test.shape)
      print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(loss), float(test_loss)))
        # outputs.append((epoch, data, recon),)
      train_list.append(float(loss))
      test_list.append(float(test_loss))       
      scheduler.step()
    return model 

#decoder = Decoder(input_len,output_len,hidden_dim, LSTM_layer)
#model = LSTM(input_len, output_len, hidden_dim, LSTM_layer, out_win, decoder, device)
#model = ConvLSTM_1step(3+param_len, hidden_dim, LSTM_layer, G, out_win, kernel_size, True, device)
if mode=='train' or mode == 'test': model = ConvLSTM_seq(7, hidden_dim, LSTM_layer, G, out_win, kernel_size, True, device, dt)
if mode=='ini': model = ConvLSTM_start(7, hidden_dim, LSTM_layer, G, out_win, kernel_size, True, device, dt)

model = model.double()
if device=='cuda':
  model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)




if model_exist:
  if mode == 'train' or mode== 'test':
    model.load_state_dict(torch.load('./lstmmodel'))
    model.eval()  
  if mode == 'ini':  
    model.load_state_dict(torch.load('./ini_lstmmodel'))
    model.eval() 
else: 
  train_list=[]
  test_list=[]
  start = time.time()
  model=train(model, num_epochs, train_loader, test_loader)
  end = time.time()
  print('training time',-start+end)
  if mode == 'train': torch.save(model.state_dict(), './lstmmodel')
  if mode == 'ini': torch.save(model.state_dict(), './ini_lstmmodel')
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
evolve_runs = num_batch*20 #num_test
frac_out = np.zeros((evolve_runs,frames,G)) ## final output
dy_out = np.zeros((evolve_runs,frames))

alone = pred_frames%out_win
pack = pred_frames-alone

param_test = param_all[num_train:,:]
param_dat = param_test[:evolve_runs,:]

seq_test = seq_all[num_train:,:,:]

if noPDE == False:
    seq_dat = seq_test[:evolve_runs,:window,:]

else: 
    ini_model = ConvLSTM_start(7, hidden_dim, LSTM_layer, G, window-1, kernel_size, True, device, dt)
    ini_model = ini_model.double()
    if device=='cuda':
       ini_model.cuda()
    init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
    print('total number of trained parameters for initialize model', init_total_params)
    ini_model.load_state_dict(torch.load('./ini_lstmmodel'))
    ini_model.eval()

    seq_1 = seq_test[:evolve_runs,[0],:]   ## this can be generated randomly
    param_dat[:,-1] = dt
    frac_new_vec = tohost( ini_model(todevice(seq_1), todevice(param_dat) )[0] ) 
    seq_dat = np.concatenate((seq_1,frac_new_vec),axis=1)
    print(frac_new_vec.shape)

## write initial windowed data to out arrays
frac_out[:,:window,:] = seq_dat[:,:,:-1]
dy_out[:,:window] = seq_dat[:,:,-1]

for i in range(0,pred_frames,out_win):
    
    param_dat[:,-1] = (i+window)*dt ## the first output time
    print('nondim time', (i+window)*dt)
    frac_new_vec = tohost( model(todevice(seq_dat), todevice(param_dat) )[0] ) 
    #print('timestep ',i)
    #print('predict',frac_new_vec/scaler_lstm[window+i])
    #print('true',frac_out_true[i,:]/scaler_lstm[window+i])
    if i>=pack:
        frac_out[:,-alone:,:] = frac_new_vec[:,:alone,:-1]
        dy_out[:,-alone:] = frac_new_vec[:,:alone,-1]
    else: 
        frac_out[:,window+i:window+i+out_win,:] = frac_new_vec[:,:,:-1]
        dy_out[:,window+i:window+i+out_win] = frac_new_vec[:,:,-1]
    #print(frac_new_vec)
    seq_dat = np.concatenate((seq_dat[:evolve_runs,out_win:,:],frac_new_vec),axis=1)
    
frac_out = frac_out/scaler_lstm[np.newaxis,:,np.newaxis] + frac_test[:evolve_runs,[0],:]
dy_out = dy_out*y_norm
dy_out[:,0] = 0
y_out = np.cumsum(dy_out,axis=-1)+y_all[num_train:num_train+evolve_runs,[0]]
#print((y_out[0,:]))
assert np.all(frac_test[:evolve_runs,0,:]==param_dat[:,:G])

sio.savemat('evolving_dat.mat',{'frac_out':frac_out,'y_out':y_out,'e_list':np.array(e_list,dtype=float),'G_list':np.array(G_list,dtype=float)})

miss_rate_param = np.zeros(num_batch)
run_per_param = int(evolve_runs/num_batch)

for batch_id in range(num_batch): 
 fname = datasets[batch_id] 
 f = h5py.File(fname, 'r')
 aseq_asse = np.asarray(f['sequence'])
 angles_asse = np.asarray(f['angles'])
 frac_asse = np.asarray(f['fractions'])
 tip_y_asse = np.asarray(f['y_t'])
 sum_miss = 0
 for plot_idx in range( run_per_param ):  # in test dataset

   data_id = plot_idx*num_batch+batch_id
   #print('seq', param_test[data_id,:])
   #frac_out_true = output_test_pt.detach().numpy()[plot_idx*pred_frames:(plot_idx+1)*pred_frames,:]
   frame_idx=plot_idx  # here the idx means the local id of the test part (last 100)
   
   alpha_true = np.asarray(f['alpha'])[frame_idx*fnx*fny:(frame_idx+1)*fnx*fny]
   aseq_test = aseq_asse[(num_train_b+frame_idx)*G:(num_train_b+frame_idx+1)*G]
   pf_angles = angles_asse[(num_train_b+frame_idx)*(G+1):(num_train_b+frame_idx+1)*(G+1)]
   tip_y = tip_y_asse[(num_train_b+frame_idx)*frames:(num_train_b+frame_idx+1)*frames]
   #print((tip_y))
   #plot_real(x,y,alpha_true,plot_idx)
   #plot_reconst(G,x,y,aseq_test,tip_y,alpha_true,frac_out[plot_idx,:,:].T,plot_idx)
   # get the parameters from dataset name
   G0 = np.float(G_list[batch_id])  #param_test[data_id,2*G+1]
   Rmax = 1 
   anis = np.float(e_list[batch_id])   #param_test[data_id,2*G]
   #plot_IO(anis,G0,Rmax,G,x,y,aseq_test,y_out[data_id,:],alpha_true,frac_out[data_id,:,:].T,window,data_id)
   #plot_IO(anis,G0,Rmax,G,x,y,aseq_test,y_out[data_id,:],alpha_true,frac_out[data_id,:,:].T,1,data_id)
   miss = miss_rate(anis,G0,Rmax,G,x,y,aseq_test,y_out[data_id,:],alpha_true,frac_out[data_id,:,:].T,window,data_id,tip_y[train_frames-1],train_frames, pf_angles)
   sum_miss = sum_miss + miss
   print('plot_id,batch_id', plot_idx, batch_id,'miss%',miss)
 miss_rate_param[batch_id] = sum_miss/run_per_param


fig, ax = plt.subplots() 

x = np.array(e_list,dtype=float)
y = np.array(G_list,dtype=float)
z = np.array(miss_rate_param,dtype=float)

cntr = ax.tricontourf(x, y, z, levels=np.linspace(0.04,0.12,9), cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
ax.plot(x, y, 'ko', ms=8)
print(x, y, z)
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.xlabel(r'$\epsilon_k$')
plt.ylabel(r'$G\ (K/ \mu m)$')
plt.title('misclassification rate')
plt.savefig('miss_rate_trunc'+str(trunc)+mode+'.png',dpi=600)

#print(miss_rate_param)