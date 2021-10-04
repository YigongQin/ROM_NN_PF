#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:34:53 2021

@author: yigongqin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from models import *
#from input1 import *
from G_E import *

def scale(t): 
    # x = 1, return 1, x = 0, return frames*beta
    return (1 - t)/dt + 1



class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        input shape: N, (2+num_param), G
        hidden sshape: N, hidden_dim, G
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]-1) // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, width, dtype=torch.float64, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, width, dtype=torch.float64, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, W or T, B, C, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state):
        '''
        input for ConvLSTM B, T, C, W 

        '''
       # if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
       #     input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
       
        b, seq_len, channel, w = input_tensor.size()

        # Implement stateful ConvLSTM
       # if hidden_state is not None:
       #     raise NotImplementedError()
       # else:
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=w)

        layer_output_list = []
        last_state_list = []

         
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                 cur_state=[h, c])
                ## output shape b, hidden_dim, w
                output_inner.append(h)
                
            ##stack every time step to form output, shape is b, t, hidden, w
            layer_output = torch.stack(output_inner, dim=1) 
            
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c]) ## 

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        ## init the hidden states for every layer
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
    
    
class Decoder(nn.Module):
    def __init__(self,input_len,output_len,hidden_dim,num_layer):
        super(Decoder, self).__init__()
        self.input_len = input_len 
        self.output_len = output_len  
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lstm_decoder = nn.LSTM(output_len,hidden_dim,num_layer,batch_first=True)    
        self.project = nn.Linear(hidden_dim, output_len)
    def forward(self,frac,hidden,cell,frac_ini,scaler):
        output, (hidden, cell) = self.lstm_decoder(frac.unsqueeze(dim=1), (hidden,cell) )
        target = self.project(output[:,-1,:])   # project last layer output to the desired shape
        target = F.relu(target+frac_ini)         # frac_ini here is necessary to keep
        frac = F.normalize(target,p=1,dim=-1)-frac_ini   # normalize the fractions
        frac = scaler.unsqueeze(dim=-1)*frac     # scale the output based on the output frame
        
        return frac, hidden, cell
# The model
class LSTM(nn.Module):
    def __init__(self,input_len,output_len,hidden_dim,num_layer,out_win,decoder,device):
        super(LSTM, self).__init__()
        self.input_len = input_len
        self.output_len = output_len  
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.out_win = out_win
        self.lstm_encoder = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)
        self.decoder = decoder
        self.device = device
      
    def forward(self, input_frac, frac_ini):
        
        output_frac = torch.zeros(input_frac.shape[0],self.out_win,self.output_len,dtype=torch.float64).to(self.device)
        ## step 1 encode the input to hidden and cell state
        encode_out, (hidden, cell) = self.lstm_encoder(input_frac)  # output range [-1,1]
        ## step 2 start with "equal vector", the last 
        frac = input_frac[:,-1,:self.output_len]  ## the ancipated output frame is 
        time_tag = input_frac[:,-1,-1]
       # param = input_1seq[:,self.output_len:] 
       ## step 3 for loop decode the time series one-by-one
        for i in range(self.out_win):
            scaler = scale(time_tag + i/(frames-1))  ## input_frac has the time information for the first output
            frac, hidden, cell = self.decoder(frac, hidden, cell, frac_ini, scaler)           
            output_frac[:,i,:] = frac
            #input_1seq[:,:self.output_len] = frac
            #param[:,-1] = param[:,-1] + 1.0/(frames-1)  ## time tag 
                        
        return output_frac

class ConvLSTM_1step(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layer, w, out_win, kernel_size, bias, device):
        super(ConvLSTM_1step, self).__init__()
        self.input_dim = input_dim  ## this input channel
        self.hidden_dim = hidden_dim  ## this output_channel
        self.num_layer = num_layer
        self.w = w
        self.out_win = out_win
        #self.lstm_encoder = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)
        self.lstm_encoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layer)
        self.project = nn.Linear(hidden_dim*w, w)## make the output channel 1
 
        self.kernel_size = kernel_size
        self.bias = bias
        self.device = device
        
    def forward(self, input_frac, frac_ini):
        
        #output_frac = torch.zeros(input_frac.shape[0],self.out_win,self.output_len,dtype=torch.float64).to(self.device)
        
        ## step 1 remap the input to the channel with gridDdim G
        ## b,t, input_len -> b,t,c,w 
        b, t, all_para  = input_frac.size()

        time_tag = input_frac[:,-1,-1]
        scaler = scale(time_tag)
        #mask = (input_frac[:,-1,:self.w]/scale(time_tag-1.0/(frames-1)).unsqueeze(dim=-1)+frac_ini>1e-3)*1
        fracs = input_frac[:,:,:self.w].unsqueeze(dim=-2)
        pf = input_frac[:,:,self.w:2*self.w].unsqueeze(dim=-2)
        param = (input_frac[:,:,2*self.w:].unsqueeze(dim=-1)) .expand(b,t,all_para-2*self.w,self.w)
        #print(fracs.shape,pf.shape,param.shape)
        input_frac = torch.cat([fracs,pf,param],dim=2)
        
        encode_out, _ = self.lstm_encoder(input_frac,None)  # output range [-1,1], None means stateless LSTM
        ## step 2 start with "equal vector", the last 
       # frac = input_frac[:,-1,:self.output_len]  ## the ancipated output frame is 
        target = self.project(encode_out[-1][:,-1,:,:].view(b,self.hidden_dim*self.w))   # project last time output b,hidden_dim, to the desired shape b,w
        
        
        target = F.relu(target+frac_ini)         # frac_ini here is necessary to keep
        frac = F.normalize(target, p=1, dim=-1)-frac_ini   # normalize the fractions
        frac = scaler.unsqueeze(dim=-1)*frac     # scale the output based on the output frame          
        #output_frac[:,0,:] = frac
        #input_1seq[:,:self.output_len] = frac
        #param[:,-1] = param[:,-1] + 1.0/(frames-1)  ## time tag 
                        
        return frac.unsqueeze(dim=1)
    
    
class ConvLSTM_seq(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layer, w, out_win, kernel_size, bias, device):
        super(ConvLSTM_seq, self).__init__()
        self.input_dim = input_dim  ## this input channel
        self.hidden_dim = hidden_dim  ## this output_channel
        self.num_layer = num_layer
        self.w = w
        self.out_win = out_win
        #self.lstm_encoder = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)
        self.lstm_encoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layer)
        self.lstm_decoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layer)
        self.project = nn.Linear(hidden_dim*w, w)## make the output channel 1
        self.project_y = nn.Linear(hidden_dim*w, 1)
    
        self.kernel_size = kernel_size
        self.bias = bias
        self.device = device
        
    def forward(self, input_seq, input_param):
        

        ## step 1 remap the input to the channel with gridDdim G
        ## b,t, input_len -> b,t,c,w 
        b, t, _  = input_seq.size()
        
        output_seq = torch.zeros(b, self.out_win, self.w+1, dtype=torch.float64).to(self.device)
        area_sum   = torch.zeros(b, self.w, dtype=torch.float64).to(self.device)
             
        frac_ini = input_param[:, :self.w]
        
        yt       = input_seq[:, :, -1:]           .view(b,t,1,1)      .expand(-1,-1, -1, self.w)
        ini      = frac_ini                       .view(b,1,1,self.w) .expand(-1, t, -1, -1)
        pf       = input_param[:, self.w:2*self.w].view(b,1,1,self.w) .expand(-1, t, -1, -1)
        param    = input_param[:, 2*self.w:]      .view(b,1,-1,1)     .expand(-1, t, -1, self.w)
        
        ## CHANNEL ORDER (7): FRAC(T), Y(T), INI, PF, P1, P2, T
        input_seq = torch.cat([input_seq[:,:,:self.w].unsqueeze(dim=-2), yt, ini, pf, param],dim=2)   
        seq_1 = input_seq[:,-1,:,:]    # the last frame

        encode_out, hidden_state = self.lstm_encoder(input_seq, None)  # output range [-1,1], None means stateless LSTM
        
        frac_old = frac_ini + seq_1[:,0,:]/ scale( seq_1[:,-1,:] - dt ) # fraction at t-1
        
        for i in range(self.out_win):
            
            encode_out, hidden_state = self.lstm_decoder(seq_1.unsqueeze(dim=1),hidden_state)
            last_time = encode_out[-1][:,-1,:,:].view(b, self.hidden_dim*self.w)
            
            y = self.project_y(last_time)    # [b,1]
            frac = self.project(last_time)   # project last time output b,hidden_dim, to the desired shape [b,w]   
            frac = F.relu(frac+frac_ini)         # frac_ini here is necessary to keep
            frac = F.normalize(frac, p=1, dim=-1)  # [b,w] normalize the fractions
            
            ## at this moment, frac is the actual fraction which can be used to calculate area
            area_sum += 0.5*( y.expand(-1,self.w) - seq_1[:,1,:] )*( frac + frac_old )
            frac_old = frac
            
            frac = scale(seq_1[:,-1,:])*( frac - frac_ini )      # [b,w] scale the output with time t    
            
            output_seq[:,i, :self.w] = frac
            output_seq[:,i, self.w:] = y
            
            ## assemble with new time-dependent variables for time t+dt: FRAC, Y, T  [b,c,w]
            
            seq_1 = torch.cat([frac.unsqueeze(dim=1), y.expand(-1,self.w).view(b,1,self.w), \
                               seq_1[:,2:-1,:], seq_1[:,-1:,:] + dt ],dim=1)

                        
        return output_seq, area_sum














