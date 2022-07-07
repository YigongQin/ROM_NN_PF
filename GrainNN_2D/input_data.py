import numpy as np
from utils import todevice, tohost, assemb_seq, divide_seq 
from torch.utils.data import Dataset, DataLoader
import h5py
import re
from math import pi
from models import ConvLSTM_start, ConvLSTM_seq, frac_norm, y_norm, area_norm
import torch



host='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def todevice(data):
    return torch.from_numpy(data).to(device)
def tohost(data):
    return data.detach().to(host).numpy()


def find_weird(frac_train, thre):

    
    diff_arr = np.absolute(np.diff(frac_train,axis=1))
    print(diff_arr.shape)
    print('maximum abs change',np.max(diff_arr))
    weird_p = np.where(diff_arr>thre)
    print('where the points are',weird_p)
    weird_sim = weird_p[0]
    weird_sim = list(set(list(weird_sim))) 
    print('weird values',diff_arr[np.where(diff_arr>thre)])
    #diff_arr[diff_arr==0.0]=np.nan
    print('the mean of the difference',np.nanmean(diff_arr))
    print('number of weird sim',len(weird_sim)) 
      
    return weird_sim

def redo_divide(frac_train, weird_sim, param_train, G, frames):
    # frac_train shape [frames,G]
    
   # for sid in [54]:
    for sid in weird_sim:
        ## redo the process of the 
      frac = frac_train[sid,:,:].squeeze()
      aseq = param_train[sid,:G].squeeze()*4.5+5.5
      #if sid ==54:
      #  print('weird sim ',sid ,'before',frac)
      #  print(aseq)
      left_coor = np.cumsum(frac[0,:])-frac[0,:]
      #print('left_coor',left_coor)
      for kt in range(1,frames):
        for j in range(1,G):
          if frac[kt,j]<1e-4 and frac[kt-1,j]>1e-4:
            left_nozero = j-1;
            while left_nozero>=0: 
                if frac[kt,left_nozero]>1e-4: break
                else: left_nozero-=1
            if left_nozero>=0 and aseq[left_nozero]==aseq[j]:
               #print("find sudden merging\n");
               all_piece = frac[kt,left_nozero]
               pre_piece = left_coor[j] - left_coor[left_nozero] 
               if pre_piece<0: pre_piece=0
               if pre_piece>all_piece: pre_piece=all_piece
               cur_piece = all_piece - pre_piece
               frac_train[sid,kt,left_nozero] = pre_piece
               frac_train[sid,kt,j] = cur_piece
               #print("correction happens, %d grain frac %f, %d grain frac %f\n" %(left_nozero,frac_train[sid,kt,left_nozero],j,frac_train[sid,kt,j]))
                      
          else:
            if j>0: left_coor[j] = (np.cumsum(frac[kt,:])-frac[kt,:])[j]
            

def check_data_quality(frac_all,param_all,y_all,G,frames):

    ### C1 check the fraction jump
    weird_sim = find_weird(frac_all, 0.25)
    '''
    refine_count=0
    while len(weird_sim)>0:
        refine_count  +=1
        redo_divide(frac_all, weird_sim, param_all, G, frames)
        weird_sim = find_weird(frac_all, 0.15)
        if refine_count ==5: break
    '''
    print(weird_sim)
    #print(param_all[weird_sim,-2:])
    ### C2 go to zero but emerge again 
    
    merge_arg = np.where( (frac_all[:,:-1,:]<1e-4)*1*(frac_all[:,1:,:]>1e-4) )
    print("renaissance", np.sum( (frac_all[:,:-1,:]<1e-4)*1*(frac_all[:,1:,:]>1e-4) ) )
    print("renaissance points", merge_arg)
    weird_sim = weird_sim #+ list(set(list(merge_arg[0])))
    #print("how emerge", frac_all[:,:-1,:][merge_arg], frac_all[:,1:,:][merge_arg])
    #print(frac_all[2489,:,:])
    
    ### C3 max and min
    print('min and max of training data', np.min(frac_all), np.max(frac_all))
    
    ## #C4 normalization
    diff_to_1 = np.absolute(np.sum(frac_all,axis=2)-1)
    #print(np.where(diff_to_1>1e-4))
    print('max diff from 1',np.max(diff_to_1))
    print('all the summation of grain fractions are 1', np.sum(diff_to_1))
    frac_all /= np.sum(frac_all,axis=2)[:,:,np.newaxis] 
    diff_to_1 = np.absolute(np.sum(frac_all,axis=2)-1)
    print('all the summation of grain fractions are 1', np.sum(diff_to_1))
    
    ### C5 check y_all for small values
    last_y = y_all[:,-1]
    print('mean and std of last y',np.mean(last_y),np.std(last_y))
    weird_y_loc = np.where(last_y<np.mean(last_y)-6*np.std(last_y))
    print('where they is small ', weird_y_loc)
    print('weird values ', last_y[weird_y_loc])
    print('weird y traj',y_all[weird_y_loc,:])
    weird_sim = weird_sim + list(weird_y_loc[0])
    return list(set(weird_sim))
    







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
         



def get_data(num_runs, num_batch, datasets, hp, param_len):
  G = hp.G
  frames = hp.frames
  all_frames = hp.all_frames
  gap = int((all_frames-1)/(frames-1))
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








## =======load data and parameters from the every simulation======


def data_setup(datasets, testsets, mode, hp, skip_check):

    batch_size = 1
    batch_train = len(datasets)
    batch_test = len(testsets)
    num_train = batch_size*batch_train
    num_test = batch_size*batch_test

    G = hp.G
    frames = hp.frames
    all_frames = hp.all_frames
    gap = int((all_frames-1)/(frames-1))


    print('==========  data information  =========')

    print('(input data) train, test', num_train, num_test)
    print('trust the data, skip check: ', skip_check)
    print('data frames: ', all_frames, 'GrainNN frames: ', frames, 'ratio: ', gap)
    print('1d grid size (number of grains): ', G)
    param_len = G + 4
    print('physical parameters: N_G orientations, G, R, e_k with length ' , param_len)
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



    frac_train, param_train, y_train, area_train, G_list, R_list, e_list = get_data(num_train, batch_train, datasets, hp, param_len)

    frac_test, param_test, y_test, area_test, _ , _ , _= get_data(num_test, batch_test, testsets, hp, param_len)
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

    param_list = [G_list, R_list, e_list, y_all]

    if not mode=='test':
       train_loader = PrepareData( data_para[0], data_para[1], data_para[2], data_para[3] )
       train_loader = DataLoader(train_loader, batch_size = 64, shuffle=True)

       return train_loader, test_loader, seq_all, param_all, param_list

    test_loader  = PrepareData(input_seq[train_sam:,:,:], output_seq[train_sam:,:,:], input_param[train_sam:,:], output_area[train_sam:,:])

    test_loader = DataLoader(test_loader, batch_size = test_sam//8, shuffle=False)


    return test_loader, seq_all, param_all, param_list



















