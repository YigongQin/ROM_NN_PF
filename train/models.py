#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:34:53 2021

@author: yigongqin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

class Decoder(nn.Module):
    def __init__(self,input_len,output_len,hidden_dim,num_layer):
        super(Decoder, self).__init__()
        self.input_len = input_len 
        self.output_len = output_len  
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lstm_decoder = nn.LSTM(output_len,hidden_dim,num_layer,batch_first=True)    
        self.project = nn.Linear(hidden_dim, output_len)
    def forward(self,frac,hidden,cell,frac_ini,scaler,mask):
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
      
    def forward(self, input_frac, frac_ini, scaler, mask):
        
        output_frac = torch.zeros(input_frac.shape[0],self.out_win,self.output_len,dtype=torch.float64).to(self.device)
        ## step 1 encode the input to hidden and cell state
        encode_out, (hidden, cell) = self.lstm_encoder(input_frac)  # output range [-1,1]
        ## step 2 start with "equal vector", the last 
        frac = input_frac[:,-1,:self.output_len]  ## the ancipated output frame is t
       # param = input_1seq[:,self.output_len:] 
       ## step 3 for loop decode the time series one-by-one
        for i in range(self.out_win):
            frac, hidden, cell = self.decoder(frac, hidden, cell, frac_ini, scaler[:,i], mask)
            
            output_frac[:,i,:] = frac
            #input_1seq[:,:self.output_len] = frac
            #param[:,-1] = param[:,-1] + 1.0/(frames-1)  ## time tag 
                        

        return output_frac
