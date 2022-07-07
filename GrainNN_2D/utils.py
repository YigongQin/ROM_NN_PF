import torch
import numpy as np


def todevice(data):
    return torch.from_numpy(data).to(device)
def tohost(data):
    return data.detach().to(host).numpy()
def assemb_seq(frac, dfrac, area, y):

    if frac.ndim==3:
       return np.concatenate((frac, dfrac, area, y[:,:,np.newaxis]), axis=-1)
    if frac.ndim==2: ## batch size = 1
       return np.concatenate((frac, dfrac, area, y[:,np.newaxis]), axis=-1)

def divide_seq(seq, G):

    if seq.ndim==3:
       return seq[:,:,:G], seq[:,:,G:2*G], seq[:,:,2*G:3*G], seq[:,:,-1]
    if seq.ndim==2:
       return seq[:,:G], seq[:,G:2*G], seq[:,2*G:3*G], seq[:,-1]