import torch
from torch.autograd import Variable as V
from torch import Tensor as T
import numpy as np
import torch.nn.functional as F

def VT(x): return V(T(x), requires_grad=False)

def three_conv(dx,dy,dz,dt, fac=1):
    #Factor for adding that single minus on the dt conv
    conv = torch.nn.Conv3d(1,4,2)
    conv.weight = torch.nn.Parameter(T(np.concatenate([dx,dy,dz,fac * dt], axis=0)))
    conv.bias = torch.nn.Parameter(T(np.array([0,0,0,0])))
    return conv

def img_derivatives(img1, img2):
    ones = np.ones((2,2,2))
    dx = (0.25 * ones * np.array([-1, 1]))[None,None,...]
    dy = (0.25 * ones * np.array([-1, 1])[:, None])[None,None,...]
    dz = 0.25 * np.stack([-np.ones((2,2)), np.ones((2,2))])[None,None,...]
    dt = ones[None, None,...]
    
    conv1 = three_conv(dx,dy,dz,dt)
    conv2 = three_conv(dx,dy,dz,dt, fac=-1)
    res =  0.5 * (conv1(VT(img1[None,...])) + conv2(VT(img2[None,...])))[0]
    #Returns a 4,50,50,50 for the 4 derivatives including time
    return F.pad(res, (1,0,1,0,1,0))

def opt_flow(dimg, r=2):
    d = dimg.shape[-1]
    x = np.ones((1,1,2,2,2))
    calc = (dimg[None, 0:3,  ...] * dimg[:,None, ...])
    conv_next = torch.nn.Conv3d(3,3,2)
    conv_next.weight = torch.nn.Parameter(T(x))
    conv_next.bias = torch.nn.Parameter(T(np.array([0])))
    
    sum_conv = torch.cat([conv_next(i[:,None,...]) for i in torch.unbind(calc, 1)], 1)
    dim = sum_conv.shape[-1]
    
    a = sum_conv
    b = a.permute(2, 3, 4, 0, 1)
    c = b[..., :-1, :].contiguous().view(-1, 3, 3)
    d = b[..., -1, :].contiguous().view(-1, 3, 1)

    inv = torch.stack([mat.inverse() for mat in torch.unbind(c, 0)])
    out = torch.bmm(inv, d)
    out = out.transpose(0,1).contiguous().view(3,dim,dim,dim)
    return out