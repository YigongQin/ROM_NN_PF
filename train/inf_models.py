#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:34:53 2021

@author: yigongqin
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import torch.nn.init as init
from typing import Callable, List, Optional, Tuple
from split_merge_reini import split_grain, merge_grain, assemb_seq, divide_seq
#from G_E import *

frac_norm = 0.06
y_norm = 1
area_norm = 10000


def scale(t,dt): 
    # x = 1, return 1, x = 0, return frames*beta
    return (1 - t)/dt + 1


class self_attention(nn.Module):

    """
    1D coarsening CNN
    Parameters:
        input_dim: Number of input channels 
        hidden_dim: Number of channels
        kernel_size: tuple, size of kernel in convolution
        bias: boolean, bias or no bias 
        Note: Will do same padding.
    Input:
        A tensor of size B, C_in, W 
    Output:
        A tensor of size B, C_out, W
    Example:
        >> y = CCNN1d(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias, device):
        super(self_attention, self).__init__()

        self.device = device
        self.w = 8  ## number of tokens/grains
        self.ds = 10
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = kernel_size[0]

        self.weight = Parameter(torch.empty((self.out_channels, self.in_channels, self.heads), dtype = torch.float64, device = device))
        self.bias = Parameter(torch.empty(out_channels,dtype = torch.float64, device = device))
        self.P = self.position()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)


    def position(self):

        #P = torch.empty((self.w, self.w, self.heads), dtype = torch.float64, device = self.device)

        idx = torch.arange(self.w, dtype = torch.float64, device = self.device)
        hid = torch.arange(self.heads, dtype = torch.float64, device = self.device) - self.heads//2
        i = idx.view(self.w,1,1)
        j = idx.view(1,self.w,1)
        h = hid.view(1,1,self.heads)

        P = self.ds*( -torch.abs(i-j+h) + self.w*( h*(i-j+h)<=0 ) )
        #print(P)
        #print(torch.softmax(P,dim=1))
        return P



    def position_2D(self):

        #P = torch.empty((self.w, self.w, self.heads), dtype = torch.float64, device = self.device)
        w_1d = int(math.sqrt(self.w))  # assume a square domain
        h_1d = int(math.sqrt(self.heads))

        idx = torch.arange(self.w, dtype = torch.float64, device = self.device)
        hid = torch.arange(self.heads, dtype = torch.float64, device = self.device)



        i = idx.view(self.w,1,1)
        j = idx.view(1,self.w,1)
        h = hid.view(1,1,self.heads)
        dist2 = lambda i, j, h: (i%w_1d - j%w_1d + h%h_1d - h_1d//2)**2 + (i//w_1d - j//w_1d + h//h_1d - h_1d//2)**2
        domain = ( (i%w_1d - j%w_1d + h%h_1d - h_1d//2)*(h%h_1d - h_1d//2)<=0 )*( (i//w_1d - j//w_1d + h//h_1d - h_1d//2)*(h//h_1d - h_1d//2)<=0 )

        P = self.ds*( -torch.sqrt(dist2(i,j,h)) + 2*w_1d*domain )


        return P


    def forward(self, input):
        '''
        input  for B, C_in, W 
        output for B, C_out, W         
        '''
        b, in_ch, w = input.size()
        # active matrix
        active = ((input[:,0,:]>1e-6)*1.0).double()
        ## [B, W, W] outer product
        #active = torch.ones((b,w),  dtype = torch.float64, device = self.device)
        I = -self.ds*(self.w+2)*( 1.0 - torch.einsum('bi, bj->bij', active, active) )

        #A = torch.eye(w, dtype =torch.float64, device = self.device).view(1,w,w,1).expand(b,w,w,self.heads)
        M = torch.ones((1,w,w), dtype = torch.float64, device = self.device) - torch.diag_embed(1.0-active)
        #print(M[0,:,:])
        A = torch.softmax( M.view(b,w,w,1)*( I.view(b,w,w,1) + self.P.view(1,w,w,self.heads) ), dim = 2 )
        #print(A[0,:,:,0])

        value = torch.einsum('biw, oih -> bowh', input, self.weight) ## [B, C, W, h]
        #output= value.view(b, self.out_channels, w)
        #output = self.linear(input.permute(0,2,1)).permute(0,2,1) 
        output = torch.sum( torch.einsum('bwkh, bokh -> bowh', A, value), dim = 3 )

        return output + self.bias.view(1,self.out_channels,1)




class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, device):
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
        self.device = device
        self.weight_ci = Parameter(torch.empty((self.hidden_dim, self.hidden_dim), dtype = torch.float64, device = device))
        self.weight_cf = Parameter(torch.empty((self.hidden_dim, self.hidden_dim), dtype = torch.float64, device = device))
        self.weight_co = Parameter(torch.empty((self.hidden_dim, self.hidden_dim), dtype = torch.float64, device = device))
        self.reset_parameters()

        ''' 
        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        '''
        self.conv = self_attention(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              bias=self.bias,
                              device=self.device)


    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)


        sc_i = torch.einsum('biw, oi -> bow', c_cur, self.weight_ci) 
        sc_f = torch.einsum('biw, oi -> bow', c_cur, self.weight_cf) 

        i = torch.sigmoid(cc_i + sc_i)
        f = torch.sigmoid(cc_f + sc_f)
        c_next = f * c_cur + i * torch.tanh(cc_g)

        sc_o = torch.einsum('biw, oi -> bow', c_next, self.weight_co) 

        o = torch.sigmoid(cc_o + sc_o)
        
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, width, dtype=torch.float64, device=self.device),
                torch.zeros(batch_size, self.hidden_dim, width, dtype=torch.float64, device=self.device))


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
                 device, batch_first=True, bias=True, return_all_layers=True):
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
        self.device = device

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          device=self.device))

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
    
    
    
def map_grain_fix(seq, frac_layer, G, G_all):
      

    expand = (G_all-G-2)//2 + 2

    args = -torch.ones((expand,G), dtype=int)

    seq_s = torch.zeros((expand,seq.shape[0],seq.shape[1],G), dtype=torch.float64).to(seq.device)

    for i in range(expand):  ## replace for loop later

        args[i,:] = torch.arange(G_all)[2*i:G+2*i]
        seq_s[i,:,:,:] = seq[:,:,args[i,:]]

    return args, seq_s

def assem_grain(v, args, G, G_all):

    new_v = torch.zeros(G_all, dtype=torch.float64).to(v.device)
    BC_l = G//2+1
    expand = args.shape[0]

    for i in range(expand):

    
        if i==0:
            new_v[args[i,:BC_l]]  = v[i, :BC_l]
        if i==expand-1:
            new_v[args[i,-BC_l:]] = v[i,-BC_l:]
        if i>0 and i<expand-1:
            new_v[args[i,G//2-1:G//2+1]] = v[i,G//2-1:G//2+1]




class ConvLSTM_seq(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layer, w, out_win, kernel_size, bias, device, dt):
        super(ConvLSTM_seq, self).__init__()
        self.input_dim = input_dim  ## this input channel
        self.hidden_dim = hidden_dim  ## this output_channel
        self.num_layer = num_layer
        self.w = w
        self.out_win = out_win
        #self.lstm_encoder = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)
        self.lstm_encoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layer[0], device)
        self.lstm_decoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layer[1], device)
        self.project = nn.Linear(hidden_dim*w, w)## make the output channel 1
        self.project_y = nn.Linear(hidden_dim*w, 1)
        self.project_a = nn.Linear(hidden_dim*w, w)

        self.kernel_size = kernel_size
        self.bias = bias
        self.device = device
        self.dt = dt
        
    def forward(self, input_seq, input_param, domain_factor):
        

        ## step 1 remap the input to the channel with gridDdim G
        ## b,t, input_len -> b,t,c,w 
        b, t, input_len  = input_seq.size()

        wa = (input_len-1)//3
        print('all g',wa)
        
        output_seq = torch.zeros(b, self.out_win, 2*wa+1, dtype=torch.float64).to(self.device)
        frac_seq = torch.zeros(b, self.out_win, wa,   dtype=torch.float64).to(self.device)
             
        frac_ini = input_param[:, :wa]
        
        yt       = input_seq[:, :, -1:]           .view(b,t,1,1)      
        ini      = frac_ini                       .view(b,1,1,wa) 
        pf       = input_param[:, wa:2*wa].view(b,1,1,wa) 
        param    = input_param[:, 2*wa:]      .view(b,1,-1,1)     
        
        ## CHANNEL ORDER (7): FRAC(T), Y(T), INI, PF, P1, P2, T
        input_seq = torch.cat([input_seq[:,:,:wa].unsqueeze(dim=-2), \
                               input_seq[:,:,wa:2*wa].unsqueeze(dim=-2), \
                               input_seq[:,:,2*wa:3*wa].unsqueeze(dim=-2), \
                               yt.expand(-1,-1, -1, wa), \
                               ini.expand(-1, t, -1, -1), \
                               pf.expand(-1, t, -1, -1), \
                               param.expand(-1, t, -1, wa)], dim=2) 



        for run in range(b):


            seq_run = input_seq[run,:,:,:]    # the last frame
            seq_1 = seq_run[-1,:,:]
            last_frac = seq_1[0,:]

            args, input_seq_s = map_grain_fix(seq_run, last_frac, self.w, wa)
            print(args)

            seq_1_s = input_seq_s[:,-1:,:,:]    # the last frame

            encode_out, hidden_state = self.lstm_encoder(input_seq_s, None)  # output range [-1,1], None means stateless LSTM
            
            #frac_old = frac_ini + seq_1[:,0,:]/ scale( seq_1[:,-1,:] - self.dt, self.dt ) # fraction at t-1
            
            for i in range(self.out_win):
                
                encode_out, hidden_state = self.lstm_decoder(seq_1_s,hidden_state)
                last_time = encode_out[-1][:,-1,:,:].view(seq_1_s.shape[0], self.hidden_dim*self.w)
                
                dy_s = F.relu(self.project_y(last_time))    # [b,1]
                darea_s = (self.project_a(last_time))    # [b,w]
                dfrac_s = self.project(last_time)/domain_factor   # project last time output b,hidden_dim, to the desired shape [b,w]   
                

                dfrac = assem_grain(dfrac_s, args, self.w, wa)
                darea = assem_grain(darea_s, args, self.w, wa)
                dy = torch.mean(dy_s, axis=0)


                frac = F.relu(dfrac+laset_frac)         # frac_ini here is necessary to keep
                frac = wa/self.w*F.normalize(frac, p=1, dim=-1)  # [b,w] normalize the fractions
                
                #active = ((frac>1e-6)*1.0).double()
                dfrac = (frac - last_frac)/frac_norm 
                last_frac = frac 

                
                output_seq[run,i, :wa] = dfrac
                output_seq[run,i, wa:2*wa] = F.relu(darea)
                output_seq[run,i, -1:] = dy
                frac_seq[run,i,:] = frac
                ## assemble with new time-dependent variables for time t+dt: FRAC, Y, T  [b,c,w]  

                seq_1 = torch.cat([frac.unsqueeze(dim=0), dfrac.unsqueeze(dim=0), darea.unsqueeze(dim=0), \
                        dy.expand(self.w).view(1,self.w), seq_1[4:-1,:], seq_1[-1:,:] + self.dt ],dim=1)

                args, seq_1_s = map_grain_fix(seq_1.view(1,-1,-1), last_frac, self.w, wa)

        return output_seq, frac_seq




class ConvLSTM_start(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layer, w, out_win, kernel_size, bias, device, dt):
        super(ConvLSTM_start, self).__init__()
        self.input_dim = input_dim  ## this input channel
        self.hidden_dim = hidden_dim  ## this output_channel
        self.num_layer = num_layer
        self.w = w
        self.out_win = out_win
        #self.lstm_encoder = nn.LSTM(input_len,hidden_dim,num_layer,batch_first=True)
        self.lstm_decoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layer[1], device)
        self.project = nn.Linear(hidden_dim*w, w)## make the output channel 1
        self.project_y = nn.Linear(hidden_dim*w, 1)
        self.project_a = nn.Linear(hidden_dim*w, w)

        self.kernel_size = kernel_size
        self.bias = bias
        self.device = device
        self.dt = dt
        
    def forward(self, input_seq, input_param, domain_factor):
        

        ## step 1 remap the input to the channel with gridDdim G
        ## b,t, input_len -> b,t,c,w 
        b, t, input_len  = input_seq.size()

        wa = (input_len-1)//3
        print('all g',wa)
        
        output_seq = torch.zeros(b, self.out_win, 2*wa+1, dtype=torch.float64).to(self.device)
        frac_seq = torch.zeros(b, self.out_win, wa,   dtype=torch.float64).to(self.device)
             
        frac_ini = input_param[:, :wa]
        
        yt       = input_seq[:, :, -1:]           .view(b,t,1,1)      
        ini      = frac_ini                       .view(b,1,1,wa) 
        pf       = input_param[:, wa:2*wa].view(b,1,1,wa) 
        param    = input_param[:, 2*wa:]      .view(b,1,-1,1)     
        
        ## CHANNEL ORDER (7): FRAC(T), Y(T), INI, PF, P1, P2, T
        input_seq = torch.cat([input_seq[:,:,:wa].unsqueeze(dim=-2), \
                               input_seq[:,:,wa:2*wa].unsqueeze(dim=-2), \
                               input_seq[:,:,2*wa:3*wa].unsqueeze(dim=-2), \
                               yt.expand(-1,-1, -1, wa), \
                               ini.expand(-1, t, -1, -1), \
                               pf.expand(-1, t, -1, -1), \
                               param.expand(-1, t, -1, wa)], dim=2) 


        for run in range(b):


            seq_run = input_seq[run,:,:,:]    # the last frame
            seq_1 = seq_run[-1,:,:]
            last_frac = seq_1[0,:]

            args, input_seq_s = map_grain_fix(seq_run, last_frac, self.w, wa)
            print(args)

            seq_1_s = input_seq_s[:,-1:,:,:]    # the last frame
            
            #frac_old = frac_ini + seq_1[:,0,:]/ scale( seq_1[:,-1,:] - self.dt, self.dt ) # fraction at t-1
            
            for i in range(self.out_win):
                
                encode_out, hidden_state = self.lstm_decoder(seq_1_s, None)
                last_time = encode_out[-1][:,-1,:,:].view(seq_1_s.shape[0], self.hidden_dim*self.w)
                
                dy_s = F.relu(self.project_y(last_time))    # [b,1]
                darea_s = (self.project_a(last_time))    # [b,w]
                dfrac_s = self.project(last_time)/domain_factor   # project last time output b,hidden_dim, to the desired shape [b,w]   
                

                dfrac = assem_grain(dfrac_s, args, self.w, wa)
                darea = assem_grain(darea_s, args, self.w, wa)
                dy = torch.mean(dy_s, axis=0)


                frac = F.relu(dfrac+laset_frac)         # frac_ini here is necessary to keep
                frac = wa/self.w*F.normalize(frac, p=1, dim=-1)  # [b,w] normalize the fractions
                
                #active = ((frac>1e-6)*1.0).double()
                dfrac = (frac - last_frac)/frac_norm 
                last_frac = frac 

                
                output_seq[run,i, :wa] = dfrac
                output_seq[run,i, wa:2*wa] = F.relu(darea)
                output_seq[run,i, -1:] = dy
                frac_seq[run,i,:] = frac
                ## assemble with new time-dependent variables for time t+dt: FRAC, Y, T  [b,c,w]  

                seq_1 = torch.cat([frac.unsqueeze(dim=0), dfrac.unsqueeze(dim=0), darea.unsqueeze(dim=0), \
                        dy.expand(self.w).view(1,self.w), seq_1[4:-1,:], seq_1[-1:,:] + self.dt ],dim=1)

                args, seq_1_s = map_grain_fix(seq_1.view(1,-1,-1), last_frac, self.w, wa)

                        
        return output_seq, frac_seq









