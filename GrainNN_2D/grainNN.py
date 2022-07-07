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

from plot_funcs import plot_IO
from torch.utils.data import Dataset, DataLoader
import glob, os, re, sys, importlib, copy
from check_data_quality import check_data_quality
from models import *
from split_merge_reini import split_grain, merge_grain, assemb_seq, divide_seq
from parameters import *


#torch.cuda.empty_cache()

mode = sys.argv[1]
all_id = int(sys.argv[2])-1
host='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_exist = False
if mode == 'test': model_exist = True
noPDE = True
plot_flag = True
skip_check = False


hp = hyperparam(mode, all_id)
frames = hp.frames
all_frames = all_frames*hp.Cl + 1
gap = int((all_frames-1)/(frames-1))

datasets = sorted(glob.glob(data_dir))
testsets = sorted(glob.glob(valid_dir))
seed_num = 35
torch.manual_seed(seed_num)
print('==========  GrainNN specification  =========')
print('2D grain microstructure evolution')
print('the mode is: ', mode, ', the model id is: ', all_id)
print('device: ',device)
print('model already exists, no training required: ', model_exist)
print('no PDE solver required, input is random: ', noPDE)
print('plot GrainNN verus PDE pointwise error: ', plot_flag)
print('\n')
print('==========  data information  =========')

print('dataset dir: ',data_dir,' data: ',len(datasets))
print('test dir: ', valid_dir,' data: ',len(testsets))
print('(input data) train, test', num_train, num_test)
print('trust the data, skip check: ', skip_check)
print('data frames: ', all_frames, 'GrainNN frames: ', frames, 'ratio: ', gap)
print('1d grid size (number of grains): ', G)
param_len = G + 4
print('physical parameters: N_G orientations, G, R, e_k with length ' , param_len)
print('\n')
print('==========  architecture  ========')
print('type -- s2s LSTM')
print('input window', hp.window,'output window', hp.out_win)
print('epochs: ', hp.epoch, 'learning rate: ', hp.lr)
print('hidden dim', hp.layer_size, 'number of layers', hp.layers)
print('convolution kernel size', hp.kernel_size)
print('torch seed', seed_num)
print('\n')



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
    e_list.append(number_list[5])
    G_list.append(number_list[6])
    R_list.append(number_list[7])
    #print(number_list[5],number_list[6],number_list[7]) 
    # compile all the datasets interleave
    for run in range(1):
      aseq = aseq_asse[run*(G+1):(run+1)*(G+1)]  # 1 to 10
      tip_y = tip_y_asse[run*all_frames:(run+1)*all_frames][::gap]
  #    Color = (aseq-3)/2        # normalize C to [-1,1]
      #Color = (aseq-5.5)/4.5
      Color = - ( 2*(aseq[1:] + pi/2)/(pi/2) - 1 )
      #print('angle sequence', Color)
      frac = (frac_asse[run*G*all_frames:(run+1)*G*all_frames]).reshape((all_frames,G))[::gap,:]
      area = (area_asse[run*G*all_frames:(run+1)*G*all_frames]).reshape((all_frames,G))[::gap,:]  # grains coalese, include frames
      #if run<1: print(frac) 
      frac_all[run*num_batch+batch_id,:,:] = frac*G/hp.G_base
      y_all[run*num_batch+batch_id,:] = tip_y 
      area_all[run*num_batch+batch_id,:,:] = area
      param_all[run*num_batch+batch_id,:G] = Color
      param_all[run*num_batch+batch_id,G] = 2*float(number_list[5])
      param_all[run*num_batch+batch_id,G+1] = 1 - np.log10(float(number_list[6]))/np.log10(100) 
      param_all[run*num_batch+batch_id,G+2] = float(number_list[7])

  return frac_all, param_all, y_all, area_all, G_list, R_list, e_list

frac_train, param_train, y_train, area_train, G_list, R_list, e_list = get_data(num_train, batch_train, datasets)

frac_test, param_test, y_test, area_test, _ , _ , _= get_data(num_test, batch_test, testsets)
#print(tip_y_asse[frames::gap])
# trained dataset need to be randomly selected:

if skip_check == False:
 weird_sim = check_data_quality(frac_train, param_train, y_train, G, frames)
else: weird_sim=[]
### divide train and validation



print('nan', np.where(np.isnan(frac_train)))
weird_sim = np.array(weird_sim)[np.array(weird_sim)<num_train]
print('throw away simulations',weird_sim)
#### delete the data in the actual training fractions and parameters
#if len(weird_sim)>0:
# frac_train = np.delete(frac_train,weird_sim,0)
# param_train = np.delete(param_train,weird_sim,0)
#num_train -= len(weird_sim) 

assert num_train==frac_train.shape[0]==param_train.shape[0]
assert num_test==frac_test.shape[0]==param_test.shape[0]
assert param_train.shape[1]==param_len
num_all = num_train + num_test

print('actual num_train',num_train)

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

seq_all = assemb_seq( frac_all, dfrac_all, darea_all, dy_all )
param_all = np.concatenate( (frac_ini, param_all), axis=1)
param_len = param_all.shape[1]
assert frac_all.shape[0] == param_all.shape[0] == y_all.shape[0] == num_all
assert param_all.shape[1] == (2*G+4)
assert seq_all.shape[2] == (3*G+1)


# Shape the inputs and outputs

num_all_traj = int(1*num_train)
all_samp = num_all*hp.S 

input_seq = np.zeros((all_samp, hp.window, 3*G+1))
input_param = np.zeros((all_samp, param_len))
output_seq = np.zeros((all_samp, hp.out_win, 2*G+1))
output_area = np.zeros((all_samp, G))
assert input_param.shape[1]==param_all.shape[1]

###### input_seq last dim seq: frac, y #######
###### input_param last dim seq: ini, phase_field, ek, G, time #######


train_sam=num_train*hp.S 
test_sam=num_test*hp.S
print('train samples', train_sam)

# Setting up inputs and outputs
# input t-hp.window to t-1
# output t to t+win_out-1
sample = 0
for run in range(num_all):
    lstm_snapshot = seq_all[run,:,:]
    if run < num_all_traj:
        end_frame = hp.S+hp.window
    else: end_frame = hp.S+hp.window
        
    for t in range(hp.window, end_frame):
        
        input_seq[sample,:,:] = lstm_snapshot[t-hp.window:t,:]        
        output_seq[sample,:,:] = lstm_snapshot[t:t+hp.out_win,G:]
        #output_seq[sample,:,:] = np.concatenate((lstm_snapshot[t:t+hp.out_win,G:2*G],lstm_snapshot[t:t+hp.out_win,-1:]),axis=-1)        
        input_param[sample,:-1] = param_all[run,:-1]  # except the last one, other parameters are independent on time
        input_param[sample,-1] = t*hp.dt 
        output_area[sample,:] = np.sum(area_all[run,t-1:t+hp.out_win-1,:],axis=0)
        
        sample = sample + 1

assert sample==input_seq.shape[0]==train_sam+test_sam
assert np.all(np.absolute(input_param[:,G:])>1e-6)

if mode=='ini': 
    input_seq[:,:,G:2*G] = 0
    input_seq[:,:,-1] = 0  
    input_param[:,:G] = input_seq[:,0,:G]  
#sio.savemat('input_trunc.mat',{'input_seq':input_seq,'input_param':input_param})




data_para = [input_seq[:train_sam,:,:], output_seq[:train_sam,:,:], \
                                           input_param[:train_sam,:], output_area[:train_sam,:]] 

if not mode=='test':
   train_loader = PrepareData( data_para[0], data_para[1], data_para[2], data_para[3] )
   train_loader = DataLoader(train_loader, batch_size = 64, shuffle=True)

test_loader  = PrepareData(input_seq[train_sam:,:,:], output_seq[train_sam:,:,:], input_param[train_sam:,:], output_area[train_sam:,:])

test_loader = DataLoader(test_loader, batch_size = test_sam//8, shuffle=False)


### prioritize the the first frame

def p(P_train):
    return (torch.ones_like(P_train[:,-1]) + 0*(P_train[:,-1]==hp.dt)).view(-1,1,1).to(device)


def train(model, num_epochs, train_loader, test_loader):
    
    #torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),lr=hp.lr) 
                                 #weight_decay=1e-5) # <--
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.5, last_epoch=-1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
 #   optimizer = AdaBound(model.parameters(),lr=learning_rate,final_lr=0.1)

    train_loss = 0
    count = 0
    for  ix, (I_train, O_train, P_train, A_train) in enumerate(train_loader):   
        count += I_train.shape[0]
        recon, area_train = model(I_train, P_train, torch.ones((I_train.shape[0], 1), dtype=torch.float64).to(device) )
        train_loss += I_train.shape[0]*float(criterion(p(P_train)*recon, p(P_train)*O_train)) #+ 0.01*hp.out_win/hp.dt*criterion(area_train, A_train)
    train_loss/=count

    test_loss = 0
    count = 0
    for  ix, (I_test, O_test, P_test, A_test) in enumerate(test_loader):      
        count += I_test.shape[0]
        pred, area_test = model(I_test, P_test, torch.ones((I_test.shape[0], 1), dtype=torch.float64).to(device))
        test_loss += I_test.shape[0]*float(criterion(p(P_test)*pred, p(P_test)*O_test)) #+ 0.01*hp.out_win/hp.dt*criterion(area_test, A_test)
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
         recon, area_train = model(I_train, P_train, torch.ones((I_train.shape[0], 1), dtype=torch.float64).to(device) )
        # loss = criterion(model(I_train, P_train), O_train)
         loss = criterion(p(P_train)*recon, p(P_train)*O_train) #+ 0.01*hp.out_win/hp.dt*criterion(area_train, A_train)

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
        pred, area_test = model(I_test, P_test, torch.ones((I_test.shape[0], 1), dtype=torch.float64).to(device))
        #test_loss = criterion(model(I_test, P_test), O_test)
        #print(criterion(pred, O_test) , hp.out_win/hp.dt*criterion(area_test, A_test))
        test_loss += I_test.shape[0]*float(criterion(p(P_test)*pred, p(P_test)*O_test)) #+ 0.01*hp.out_win/hp.dt*criterion(area_test, A_test)
 
      test_loss/=count
      print('Epoch:{}, Train loss:{:.6f}, valid loss:{:.6f}'.format(epoch+1, float(train_loss), float(test_loss)))
 
      train_list.append(float(loss))
      test_list.append(float(test_loss))       
      scheduler.step()

    return model 


if mode=='train' or mode == 'test': model = ConvLSTM_seq(10, hp, True, device)
if mode=='ini': model = ConvLSTM_start(10, hp, True, device)

model = model.double()
if device=='cuda':
  model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total number of trained parameters ', pytorch_total_params)



if model_exist==False: 
  train_list=[]
  test_list=[]
  start = time.time()
  model=train(model, num_epochs, train_loader, test_loader)
  end = time.time()
  print('training time',-start+end)
  if mode == 'train': torch.save(model.state_dict(), './lstmmodel'+str(all_id))
  if mode == 'ini': torch.save(model.state_dict(), './ini_lstmmodel'+str(all_id))
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

if mode!='test' and model_exist==False: sio.savemat('loss_curve_mode'+mode+'.mat',{'train':train_list,'test':test_list})


#left_grains[:,0,:] = np.cumsum(frac_out[:,0,:], axis=-1) - frac_out[:,0,:]

def network_inf(seq_out,param_dat, model, ini_model, hp):
    if noPDE == False:
        seq_dat = seq_test[:evolve_runs,:hp.window,:]

        frac_out[:,:hp.window,:] = seq_dat[:,:,:G]
        dy_out[:,:hp.window] = seq_dat[:,:,-1]
        darea_out[:,:hp.window,:] = seq_dat[:,:,2*G:3*G]

        param_dat, seq_dat, expand, left_coors = split_grain(param_dat, seq_dat, hp.G_base, G)
    else: 


        seq_1 = seq_out[:,[0],:]   ## this can be generated randomly
        seq_1[:,:,-1]=0
        seq_1[:,:,G:2*G]=0
        print('sample', seq_1[0,0,:])

        param_dat_s, seq_1_s, expand, domain_factor, left_coors = split_grain(param_dat, seq_1, hp.G_base, G)

        param_dat_s[:,-1] = hp.dt
        domain_factor = hp.Cl*domain_factor
        seq_1_s[:,:,2*hp.G_base:3*hp.G_base] /= hp.Cl

        output_model = ini_model(todevice(seq_1_s), todevice(param_dat_s), todevice(domain_factor) )
        dfrac_new = tohost( output_model[0] ) 
        frac_new = tohost(output_model[1])

        dfrac_new[:,:,hp.G_base:2*hp.G_base] *= hp.Cl

        #frac_out[:,1:hp.window,:], dy_out[:,1:hp.window], darea_out[:,1:hp.window,:], left_grains[:,1:hp.window,:] \
        seq_out[:,1:hp.window,:], left_grains[:,1:hp.window,:] \
            = merge_grain(frac_new, dfrac_new, hp.G_base, G, expand, domain_factor, left_coors)

        seq_dat = seq_out[:,:hp.window,:]
        seq_dat_s = np.concatenate((seq_1_s,np.concatenate((frac_new, dfrac_new), axis = -1)),axis=1)
        if mode != 'ini':
          seq_dat[:,0,-1] = seq_dat[:,1,-1]
          seq_dat[:,0,G:2*G] = seq_dat[:,1,G:2*G] 



    ## write initial windowed data to out arrays

    #print('the sub simulations', expand)
    alone = hp.pred_frames%hp.out_win
    pack = hp.pred_frames-alone

    for i in range(0,hp.pred_frames,hp.out_win):
        
        time_i = i
        if hp.dt*(time_i+hp.window+hp.out_win-1)>1: 
            time_i = int(1/hp.dt)-(hp.window+hp.out_win-1)
        ## you may resplit the grains here

        param_dat_s, seq_dat_s, expand, domain_factor, left_coors = split_grain(param_dat, seq_dat, hp.G_base, G)

        param_dat_s[:,-1] = (time_i+hp.window)*hp.dt ## the first output time
        print('nondim time', (time_i+hp.window)*hp.dt)

        domain_factor = hp.Cl*domain_factor
        seq_dat_s[:,:,2*hp.G_base:3*hp.G_base] /= hp.Cl

        output_model = model(todevice(seq_dat_s), todevice(param_dat_s), todevice(domain_factor)  )
        dfrac_new = tohost( output_model[0] ) 
        frac_new = tohost(output_model[1])

        dfrac_new[:,:,hp.G_base:2*hp.G_base] *= hp.Cl

        if i>=pack and mode!='ini':
            seq_out[:,-alone:,:], left_grains[:,-alone:,:] \
            = merge_grain(frac_new[:,:alone,:], dfrac_new[:,:alone,:], hp.G_base, G, expand, domain_factor, left_coors)
        else: 
            seq_out[:,hp.window+i:hp.window+i+hp.out_win,:], left_grains[:,hp.window+i:hp.window+i+hp.out_win,:] \
            = merge_grain(frac_new, dfrac_new, hp.G_base, G, expand, domain_factor, left_coors)
        
        seq_dat = np.concatenate((seq_dat[:,hp.out_win:,:], seq_out[:,hp.window+i:hp.window+i+hp.out_win,:]),axis=1)
       

    frac_out, dfrac_out, darea_out, dy_out = divide_seq(seq_out, G)
    frac_out *= hp.G_base/G
    dy_out = dy_out*y_norm
    dy_out[:,0] = 0
    y_out = np.cumsum(dy_out,axis=-1)+y_all[num_train:num_train+evolve_runs,[0]]

    area_out = darea_out*area_norm
    return frac_out, y_out, area_out


def ensemble(seq_out, param_dat, inf_model_list):

    Nmodel = len(inf_model_list)


    frac_out = np.zeros((Nmodel,evolve_runs,frames,G)) ## final output
    area_out = np.zeros((Nmodel,evolve_runs,frames,G)) ## final output
    y_out = np.zeros((Nmodel,evolve_runs,frames))
    for i in range(Nmodel):

        seq_i = copy.deepcopy(seq_out)
        param_i = copy.deepcopy(param_dat)
        all_id = inf_model_list[i]
        hp = hyperparam('test', all_id)

        model = ConvLSTM_seq(10, hp, True, device)
        model = model.double()
        if device=='cuda':
            model.cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('total number of trained parameters ', pytorch_total_params)
        model.load_state_dict(torch.load(model_dir+'/lstmmodel'+str(all_id)))
        model.eval()  


        ini_model = ConvLSTM_start(10, hp, True, device)
        ini_model = ini_model.double()
        if device=='cuda':
           ini_model.cuda()
        init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
        print('total number of trained parameters for initialize model', init_total_params)
        ini_model.load_state_dict(torch.load(model_dir+'/ini_lstmmodel'+str(all_id)))
        ini_model.eval()


        frac_out[i,:,:,:], y_out[i,:,:], area_out[i,:,:,:] = network_inf(seq_i, param_i, model, ini_model, hp)

 
    return np.mean(frac_out,axis=0), np.mean(y_out,axis=0), np.mean(area_out,axis=0)


evolve_runs = num_test #num_test

seq_out = np.zeros((evolve_runs,frames,3*G+1))
left_grains = np.zeros((evolve_runs,frames,G))

seq_out[:,0,:] = seq_all[num_train:,0,:]
param_dat = param_all[num_train:,:]

if mode!='test':

    if model_exist:
      if mode == 'train' :
        model.load_state_dict(torch.load(model_dir+'/lstmmodel'+str(all_id)))
        model.eval()  
      if mode == 'ini':  
        model.load_state_dict(torch.load(model_dir+'/ini_lstmmodel'+str(all_id)))
        model.eval() 

    ini_model = ConvLSTM_start(10, hp, True, device)
    ini_model = ini_model.double()
    if device=='cuda':
       ini_model.cuda()
    init_total_params = sum(p.numel() for p in ini_model.parameters() if p.requires_grad)
    print('total number of trained parameters for initialize model', init_total_params)
    ini_model.load_state_dict(torch.load(model_dir+'/ini_lstmmodel'+str(all_id)))
    ini_model.eval()

    frac_out, y_out, area_out = network_inf(seq_out, param_dat, model, ini_model, hp)

if mode=='test':
    inf_model_list = [42,24, 69,71]
    nn_start = time.time()
    frac_out, y_out, area_out = ensemble(seq_out, param_dat, inf_model_list)
    nn_end = time.time()
    print('===network inference time %f seconds =====', nn_end-nn_start)


'''
seq_reverse = copy.deepcopy(seq_out)
param_reverse = copy.deepcopy(param_dat)
seq_reverse [:,:,:G]      = np.flip(seq_out[:,:,:G],axis=-1)
seq_reverse [:,:,G:2*G]   = np.flip(seq_out[:,:,G:2*G],axis=-1)
seq_reverse [:,:,2*G:3*G] = np.flip(seq_out[:,:,2*G:3*G],axis=-1)

param_reverse [:,:G]      = np.flip(param_dat[:,:G],axis=-1)
param_reverse [:,G:2*G]   = -np.flip(param_dat[:,G:2*G],axis=-1)

print(seq_reverse[0,0,:],seq_out[0,0,:])
print(param_reverse[0,:],param_dat[0,:])

frac_out_r, y_out_r, area_out_r = network_inf(seq_reverse, param_reverse)
frac_out_r= np.flip(frac_out_r,axis=-1)
area_out_r= np.flip(area_out_r,axis=-1)


frac_out_f, y_out_f, area_out_f = network_inf(seq_out, param_dat)

frac_out = 0.5*(frac_out_f+frac_out_r)
y_out = 0.5*(y_out_f+y_out_r)
area_out = 0.5*(area_out_f+area_out_r)
'''

#darea_out[:,0,:] = 0
#area_out = np.cumsum(darea_out,axis=1)+area_all[num_train:num_train+evolve_runs,[0],:]
#print((y_out[0,:]))

dice = np.zeros((num_test,G))
miss_rate_param = np.zeros(num_test)
run_per_param = int(evolve_runs/batch_test)
if run_per_param <1: run_per_param = 1

if mode == 'test': valid_train = True
else: valid = False
valid_train = True
if valid_train:
  aseq_test = np.arange(G)+1
  for batch_id in range(batch_test): 
   fname = testsets[batch_id] 
   f = h5py.File(fname, 'r')
 
   angles_asse = np.asarray(f['angles'])
   alpha_asse = np.asarray(f['alpha'])
   G0 = float(G_list[batch_id]) 
   Rmax = float(R_list[batch_id]) 
   anis = float(e_list[batch_id])  
   for plot_idx in range( run_per_param ):  # in test dataset

     data_id = plot_idx*batch_test+batch_id
     #print('seq', param_test[data_id,:])
     pf_angles = angles_asse[plot_idx*(G+1):(plot_idx+1)*(G+1)]
     pf_angles[1:] = pf_angles[1:]*180/pi + 90     
     for dat_frames in [all_frames-1]:
         
         inf_frames = dat_frames//gap + 1
         extra_time = dat_frames/gap - dat_frames//gap
         area = area_out[data_id,inf_frames-1,:]
         if extra_time>0: area += extra_time*(area_out[data_id,inf_frames,:]-area_out[data_id,inf_frames-1,:])

         frame_idx = all_frames*plot_idx + dat_frames  ## uncomment this line if there are more than one frame in dat file
         frame_idx = plot_idx

         alpha_true = alpha_asse[frame_idx*fnx*fny:(frame_idx+1)*fnx*fny]
         alpha_true = np.reshape(alpha_true,(fnx,fny),order='F')[1:-1,1:-1]   

         miss_rate_param[data_id], dice[data_id,:] = plot_IO(anis,G0,Rmax,G,x,y,aseq_test,pf_angles,alpha_true,\
            y_out[data_id,:],frac_out[data_id,:,:].T, area, inf_frames, extra_time, plot_flag,data_id)

         print('realization id, param id', plot_idx, batch_id, 'frame id' , dat_frames, 'miss%',miss_rate_param[data_id])



#fig, ax = plt.subplots() 

x = np.array(e_list,dtype=float)
y = np.array(G_list,dtype=float)
z = np.array(R_list,dtype=float)
u = np.array(miss_rate_param,dtype=float)

print(x)
print(y)
print(z)
print(u)
print('for model ', all_id, 'the mean error', np.mean(u))

ave_err = np.mean(u)

if mode == 'test':
  print('all id', all_id, 'layer_size', hp.layer_size, 'learning_rate', hp.lr, \
    'num_layers', hp.layers, 'frames', frames, 'out win', hp.out_win, 'err', ave_err)
else:
      print('all id', all_id, 'layer_size', hp.layer_size, 'learning_rate', hp.lr, \
    'num_layers', hp.layers, 'frames', frames, 'out win', hp.out_win, 'err', ave_err, 'time', -start+end)
sio.savemat('2D_train'+str(num_train)+'_test'+str(num_test)+'_mode_'+mode+'_id_'+str(all_id)+'err'+str('%1.3f'%ave_err)+'.mat',{'frac_out':frac_out,'y_out':y_out,'area_out':area_out,'e':x,'G':y,'R':z,'err':u,'dice':dice,\
  'seq_all':seq_all,'param_all':param_all,'layer_size':hp.layer_size, 'learning_rate':hp.lr, 'num_layers':hp.layers, 'frames':frames, \
  'frac_true':frac_test,'y_true':y_test,'area_true':area_test})


