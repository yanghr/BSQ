import math

import numpy as np
import torch
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


############################################################################################################################################################

## Straight Through Estimator, modified from https://github.com/zjysteven/bitslice_sparsity/blob/master/mnist/pretrain.py#L181

############################################################################################################################################################

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit):
        if w is None:
            return None
        if bit==0:
            weight = w*0
        else:
            S = torch.max(torch.abs(w))
            if S==0:
                weight = w*0
            else:
                step = 2 ** (bit)-1
                R = torch.round(torch.abs(w) * step / S)/step
                weight =  S * R * torch.sign(w)
        return weight

    @staticmethod
    def backward(ctx, g):
        return g, None
        
class bit_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit, zero):
        if w is None:
            return None
        if zero or bit==0:
            weight = w*0
        else:
            w = torch.where(w > 1, torch.full_like(w, 1), w)
            w = torch.where(w < -1, torch.full_like(w, -1), w)
            step = 2 ** (bit)-1
            R = torch.round(torch.abs(w) * step)/step
            weight =  R * torch.sign(w)
        return weight

    @staticmethod
    def backward(ctx, g):
        return g, None, None


############################################################################################################################################################

## Fully-connected Layer, modified from https://github.com/mightydeveloper/Deep-Compression-PyTorch/blob/master/net/prune.py

############################################################################################################################################################

class BitLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """
    def __init__(self, in_features, out_features, Nbits=8, bias=True, bin=False):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Nbits = Nbits
        ex = np.arange(Nbits-1, -1, -1)
        self.exps = torch.Tensor((2**ex)/(2**(self.Nbits)-1)).float()
        self.bNbits = Nbits
        self.bexps = torch.Tensor((2**ex)/(2**(self.bNbits)-1)).float()
        self.bin = bin
        self.total_weight = out_features*in_features
        self.total_bias = out_features
        self.zero=False
        self.bzero=False
        self.ft = False
        if self.bin:
            self.pweight = Parameter(torch.Tensor(out_features, in_features, Nbits))
            self.nweight = Parameter(torch.Tensor(out_features, in_features, Nbits))
            self.scale = Parameter(torch.Tensor(1))
            if bias:
                self.pbias = Parameter(torch.Tensor(out_features, Nbits))
                self.nbias = Parameter(torch.Tensor(out_features, Nbits))
                self.biasscale = Parameter(torch.Tensor(1))
            else:
                self.register_parameter('pbias', None)
                self.register_parameter('nbias', None)
                self.register_parameter('biasscale', None)
            self.bin_reset_parameters()
            # book keeping
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            # book keeping
            self.register_parameter('pweight', None)
            self.register_parameter('nweight', None)
            self.register_parameter('scale', None)
            self.register_parameter('pbias', None)
            self.register_parameter('nbias', None)
            self.register_parameter('biasscale', None)
            

    def reset_parameters(self):
    # For float model
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def ini2bit(self, ini, b=False, ft=False):
    # For binary model
        if ft:
            S = 1.0
        else:
            S = torch.max(torch.abs(ini))
        if S==0:
            if b:
                self.pbias.data.fill_(0)
                self.nbias.data.fill_(0)
            else:
                self.pweight.data.fill_(0)
                self.nweight.data.fill_(0)
            return
            
        inip = torch.where(ini > 0, ini, torch.full_like(ini, 0))
        inin = torch.where(ini <= 0, -ini, torch.full_like(ini, 0))
        if b:
            step = 2 ** (self.bNbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.biasscale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.bNbits):
                ex = 2**(self.bNbits-i-1)
                self.pbias.data[:,i] = torch.floor(Rp/ex)
                self.nbias.data[:,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex
        else:
            step = 2 ** (self.Nbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.scale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.Nbits):
                ex = 2**(self.Nbits-i-1)
                self.pweight.data[...,i] = torch.floor(Rp/ex)
                self.nweight.data[...,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex

    def bin_reset_parameters(self):
    # For binary model
        stdv = 1. / math.sqrt(self.pweight.size(1))
        ini_w = torch.Tensor(self.out_features, self.in_features).uniform_(-stdv, stdv)
        self.ini2bit(ini_w)
        if self.pbias is not None:
            ini_b = torch.Tensor(self.out_features).uniform_(-stdv, stdv)
            self.ini2bit(ini_b, b=True)


    def to_bin(self):
        if self.bin:
            return
        else:
            self.bin = True
            self.ft = False
            if self.Nbits==0:
                self.Nbits=1
                self.weight.data = self.weight.data*0
                self.zero=True
            if self.bNbits==0:
                self.bNbits=1
                self.bias.data = self.bias.data*0
                self.bzero=True
            self.pweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
            self.nweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
            self.scale = Parameter(self.weight.data.new_zeros(1))
            self.ini2bit(self.weight.data)
            self.weight = None
            if self.bias is not None:
                self.pbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
                self.nbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
                self.biasscale = Parameter(self.bias.data.new_zeros(1))
                self.ini2bit(self.bias.data, b=True)
                self.bias = None
    
    def to_float(self):
        if self.bin:
            self.bin = False
            self.ft = False
            self.weight = Parameter(self.pweight.data.new_zeros(self.out_features, self.in_features))
            dev = self.pweight.device
            weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight,dim=2) * self.scale
            if self.Nbits==1 and (np.count_nonzero(self.weight.data.cpu().numpy())==0):
                self.Nbits=0
            self.pweight = None
            self.nweight = None
            self.scale = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias,dim=1) * self.biasscale
                if self.bNbits==1 and (np.count_nonzero(self.bias.data.cpu().numpy())==0):
                    self.bNbits=0
                self.pbias = None
                self.nbias = None
                self.biasscale = None
        else:
            return
    
    def finetune_to_bit(self):  
        self.ft=False
        self.bin = True
        if self.Nbits==0:
            self.Nbits=1
            self.weight.data = self.weight.data*0
            self.zero=True
        if self.bNbits==0:
            self.bNbits=1
            self.bias.data = self.bias.data*0
            self.bzero=True
        self.pweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
        self.nweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
        w = self.weight.data
        w = torch.where(w > 1, torch.full_like(w, 1), w)
        w = torch.where(w < -1, torch.full_like(w, -1), w)
        self.ini2bit(w,ft=True)
        self.weight = None
        if self.bias is not None:
            self.pbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            self.nbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            w = self.bias.data
            w = torch.where(w > 1, torch.full_like(w, 1), w)
            w = torch.where(w < -1, torch.full_like(w, -1), w)
            self.ini2bit(w,b=True,ft=True)
            self.bias = None
    
    def to_finetune(self):
        if self.bin:
            self.bin = False
            self.ft=True
            self.weight = Parameter(self.pweight.data.new_zeros(self.out_features, self.in_features))
            dev = self.pweight.device
            weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight,dim=2)
            if self.Nbits==1 and (np.count_nonzero(self.weight.data.cpu().numpy())==0):
                self.Nbits=0
            self.pweight = None
            self.nweight = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias,dim=1)
                if self.bNbits==1 and (np.count_nonzero(self.bias.data.cpu().numpy())==0):
                    self.bNbits=0
                self.pbias = None
                self.nbias = None
        else:
            return

    def forward(self, input):
        if self.bin:
            dev = self.pweight.device
            weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
            weight = bit_STE.apply(torch.sum(weight,dim=2), self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                bias = bit_STE.apply(torch.sum(bias,dim=1), self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return F.linear(input, weight, bias)
        elif self.ft:
            weight = bit_STE.apply(self.weight, self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = bit_STE.apply(self.bias, self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, self.weight, self.bias)
            #return F.linear(input, STE.apply(self.weight, self.Nbits), STE.apply(self.bias,self.bNbits))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def quant(self):
    # For binary model
        ## Re-quantize the binary part of the weight, keep scale unchanged
        dev = self.pweight.device
        ## Quantize weight
        weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
        weight = torch.sum(weight,dim=2)
        inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
        inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
        step = 2 ** (self.Nbits)-1
        Rp = torch.round(inip * step)
        Rn = torch.round(inin * step)
        Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
        Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
        for i in range(self.Nbits):
            ex = 2**(self.Nbits-i-1)
            self.pweight.data[...,i] = torch.floor(Rp/ex)
            self.nweight.data[...,i] = torch.floor(Rn/ex)
            Rp = Rp-torch.floor(Rp/ex)*ex
            Rn = Rn-torch.floor(Rn/ex)*ex
        ## Quantize bias
        if self.pbias is not None:
            weight = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
            weight = torch.sum(weight,dim=1)
            inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
            inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
            step = 2 ** (self.bNbits)-1
            Rp = torch.round(inip * step)
            Rn = torch.round(inin * step)
            Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
            Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
            for i in range(self.bNbits):
                ex = 2**(self.bNbits-i-1)
                self.pbias.data[:,i] = torch.floor(Rp/ex)
                self.nbias.data[:,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex
                
    def print_stat(self):
    # For binary model
        weight = self.pweight.data.cpu().numpy()-self.nweight.data.cpu().numpy()
        total_weight = np.prod(weight.shape)/self.Nbits
        nonz_weight = [np.count_nonzero(weight[...,i])*100 for i in range(self.Nbits)]
        print('Weight: '+np.array2string(nonz_weight/total_weight, separator='%, ', formatter={'float_kind':lambda x: "%6.2f" % x}).strip('[]')+'%')
        if self.pbias is not None:
            bia = self.pbias.data.cpu().numpy()-self.nbias.data.cpu().numpy()
            total_weight = np.prod(bia.shape)/self.bNbits
            nonz_weight = [np.count_nonzero(bia[:,i])*100 for i in range(self.bNbits)]
            print('Bias:   '+np.array2string(nonz_weight/total_weight, separator='%, ', formatter={'float_kind':lambda x: "%6.2f" % x}).strip('[]')+'%')
        

    def L1reg(self,reg):
    # For binary model
        param = torch.cat((self.pweight,self.nweight),0)
        total_weight = self.total_weight*self.Nbits
        reg += total_weight*torch.sum(torch.sqrt(1e-8+torch.sum(param**2,(0,1))))
        if self.pbias is not None:
            param = torch.cat((self.pbias,self.nbias),0)
            total_bias = self.total_bias*self.bNbits
            reg += total_bias*torch.sum(torch.sqrt(1e-8+torch.sum(param**2,0)))
        return reg

############################################################################################################################################################

## Convolutional Layer, modified from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html

############################################################################################################################################################

class Bit_ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'Nbits']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, Nbits=8, bin=False):
        super(Bit_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.Nbits = Nbits
        ex = np.arange(Nbits-1, -1, -1)
        self.exps = torch.Tensor((2**ex)/(2**(self.Nbits)-1)).float()
        self.bNbits = Nbits
        self.bexps = torch.Tensor((2**ex)/(2**(self.bNbits)-1)).float()
        self.zero=False
        self.bzero=False
        self.ft=False
        self.bin = bin
        if self.bin:
            if transposed:
                self.pweight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size, Nbits))
                self.nweight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size, Nbits))
                self.scale = Parameter(torch.Tensor(1))
            else:
                self.pweight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size, Nbits))
                self.nweight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size, Nbits))
                self.scale = Parameter(torch.Tensor(1))
                
            if bias:
                self.pbias = Parameter(torch.Tensor(out_channels, Nbits))
                self.nbias = Parameter(torch.Tensor(out_channels, Nbits))
                self.biasscale = Parameter(torch.Tensor(1))
            else:
                self.register_parameter('pbias', None)
                self.register_parameter('nbias', None)
                self.register_parameter('biasscale', None)
            self.bin_reset_parameters()
            # book keeping
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            if transposed:
                self.weight = Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            # book keeping
            self.register_parameter('pweight', None)
            self.register_parameter('nweight', None)
            self.register_parameter('scale', None)
            self.register_parameter('pbias', None)
            self.register_parameter('nbias', None)
            self.register_parameter('biasscale', None)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            

    def ini2bit(self, ini, b=False, ft=False):
    # For binary model
        if ft:
            S = 1.0
        else:
            S = torch.max(torch.abs(ini))
        if S==0:
            if b:
                self.pbias.data.fill_(0)
                self.nbias.data.fill_(0)
            else:
                self.pweight.data.fill_(0)
                self.nweight.data.fill_(0)
            return
            
        inip = torch.where(ini > 0, ini, torch.full_like(ini, 0))
        inin = torch.where(ini <= 0, -ini, torch.full_like(ini, 0))
        if b:
            step = 2 ** (self.bNbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.biasscale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.bNbits):
                ex = 2**(self.bNbits-i-1)
                self.pbias.data[:,i] = torch.floor(Rp/ex)
                self.nbias.data[:,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex
        else:
            step = 2 ** (self.Nbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.scale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.Nbits):
                ex = 2**(self.Nbits-i-1)
                self.pweight.data[...,i] = torch.floor(Rp/ex)
                self.nweight.data[...,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex

    def bin_reset_parameters(self):
        ini_w = torch.full_like(self.pweight[...,0], 0)
        init.kaiming_uniform_(ini_w, a=math.sqrt(5))
        self.ini2bit(ini_w)
        if self.pbias is not None:
            #stdv = 1. / math.sqrt(self.pweight.size(1))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.pweight)
            stdv = 1 / math.sqrt(fan_in)
            ini_b = torch.Tensor(self.out_channels).uniform_(-stdv, stdv)
            self.ini2bit(ini_b, b=True)
    
    def to_bin(self):
        if self.bin:
            return
        else:
            self.bin = True
            self.ft = False
            if self.Nbits==0:
                self.Nbits=1
                self.weight.data = self.weight.data*0
                self.zero=True
            if self.bNbits==0:
                self.bNbits=1
                self.bias.data = self.bias.data*0
                self.bzero=True
            if self.transposed:
                self.pweight = Parameter(self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size, self.Nbits))
                self.nweight = Parameter(self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size, self.Nbits))
            else:
                self.pweight = Parameter(self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size, self.Nbits))
                self.nweight = Parameter(self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size, self.Nbits))
            self.scale = Parameter(self.weight.data.new_zeros(1))
            #print(self.weight.data.shape)
            self.ini2bit(self.weight.data)
            self.weight = None
            if self.bias is not None:
                self.pbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
                self.nbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
                self.biasscale = Parameter(self.bias.data.new_zeros(1))
                self.ini2bit(self.bias.data, b=True)
                self.bias = None
    
    def to_float(self):
        if self.bin:
            self.bin = False
            self.ft = False
            if self.transposed:
                self.weight = Parameter(self.pweight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size))
            else:
                self.weight = Parameter(self.pweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
            dev = self.pweight.device
            weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight,dim=-1) * self.scale
            if self.Nbits==1 and (np.count_nonzero(self.weight.data.cpu().numpy())==0):
                self.Nbits=0
            self.pweight = None
            self.nweight = None
            self.scale = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias,dim=1) * self.biasscale
                if self.bNbits==1 and np.count_nonzero(self.bias.data.cpu().numpy())==0:
                    self.bNbits=0
                self.pbias = None
                self.nbias = None
                self.biasscale = None
        else:
            return
            
    def finetune_to_bit(self):  
        self.ft=False
        self.bin = True
        if self.Nbits==0:
            self.Nbits=1
            self.weight.data = self.weight.data*0
            self.zero=True
        if self.bNbits==0:
            self.bNbits=1
            self.bias.data = self.bias.data*0
            self.bzero=True
        if self.transposed:
            self.pweight = Parameter(self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size, self.Nbits))
            self.nweight = Parameter(self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size, self.Nbits))
        else:
            self.pweight = Parameter(self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size, self.Nbits))
            self.nweight = Parameter(self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size, self.Nbits))
        w = self.weight.data
        w = torch.where(w > 1, torch.full_like(w, 1), w)
        w = torch.where(w < -1, torch.full_like(w, -1), w)
        self.ini2bit(w,ft=True)
        self.weight = None
        if self.bias is not None:
            self.pbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            self.nbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            w = self.bias.data
            w = torch.where(w > 1, torch.full_like(w, 1), w)
            w = torch.where(w < -1, torch.full_like(w, -1), w)
            self.ini2bit(w,b=True,ft=True)
            self.bias = None     
    
    def to_finetune(self):
        if self.bin:
            self.bin = False
            self.ft=True
            if self.transposed:
                self.weight = Parameter(self.pweight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size))
            else:
                self.weight = Parameter(self.pweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
            dev = self.pweight.device
            weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight,dim=-1)
            if self.Nbits==1 and (np.count_nonzero(self.weight.data.cpu().numpy())==0):
                self.Nbits=0
            self.pweight = None
            self.nweight = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias,dim=1)
                if self.bNbits==1 and (np.count_nonzero(self.bias.data.cpu().numpy())==0):
                    self.bNbits=0
                self.pbias = None
                self.nbias = None
        else:
            return
           
    def print_stat(self):
        weight = self.pweight.data.cpu().numpy()-self.nweight.data.cpu().numpy()
        total_weight = np.prod(weight.shape)/self.Nbits
        nonz_weight = [np.count_nonzero(weight[...,i])*100 for i in range(self.Nbits)]
        #print(self.scale.data.cpu().numpy())
        print('Weight: '+np.array2string(nonz_weight/total_weight, separator='%, ', formatter={'float_kind':lambda x: "%6.2f" % x}).strip('[]')+'%')
        if self.pbias is not None:
            bia = self.pbias.data.cpu().numpy()-self.nbias.data.cpu().numpy()
            total_weight = np.prod(bia.shape)/self.bNbits
            nonz_weight = [np.count_nonzero(bia[:,i])*100 for i in range(self.bNbits)]
            print('Bias:   '+np.array2string(nonz_weight/total_weight, separator='%, ', formatter={'float_kind':lambda x: "%6.2f" % x}).strip('[]')+'%')

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'



class BitConv2d(Bit_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', Nbits=8, bin=False):
        
        self.total_weight = (in_channels//groups)*out_channels*kernel_size*kernel_size
        self.total_bias = out_channels
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BitConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, Nbits, bin)

    def quant(self):
        ## Re-quantize the binary part of the weight, keep scale unchanged
        dev = self.pweight.device
        ## Quantize weight
        weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
        weight = torch.sum(weight,dim=4)
        inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
        inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
        step = 2 ** (self.Nbits)-1
        Rp = torch.round(inip * step)
        Rn = torch.round(inin * step)
        Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
        Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
        for i in range(self.Nbits):
            ex = 2**(self.Nbits-i-1)
            self.pweight.data[...,i] = torch.floor(Rp/ex)
            self.nweight.data[...,i] = torch.floor(Rn/ex)
            Rp = Rp-torch.floor(Rp/ex)*ex
            Rn = Rn-torch.floor(Rn/ex)*ex
        ## Quantize bias
        if self.pbias is not None:
            weight = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
            weight = torch.sum(weight,dim=1)
            inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
            inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
            step = 2 ** (self.bNbits)-1
            Rp = torch.round(inip * step)
            Rn = torch.round(inin * step)
            Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
            Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
            for i in range(self.bNbits):
                ex = 2**(self.bNbits-i-1)
                self.pbias.data[:,i] = torch.floor(Rp/ex)
                self.nbias.data[:,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if self.bin:
            dev = self.pweight.device
            weight = torch.mul((self.pweight-self.nweight), self.exps.to(dev))
            weight = bit_STE.apply(torch.sum(weight,dim=4), self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                bias = bit_STE.apply(torch.sum(bias,dim=1), self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        elif self.ft:
            weight = bit_STE.apply(self.weight, self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = bit_STE.apply(self.bias, self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        else:
            return self.conv2d_forward(input, self.weight, self.bias)
            #return self.conv2d_forward(input, STE.apply(self.weight, self.Nbits), self.bias)
        
    def L1reg(self,reg):
        param = torch.cat((self.pweight,self.nweight),0)
        total_weight = self.total_weight*self.Nbits
        reg += total_weight*torch.sum(torch.sqrt(1e-8+torch.sum(param**2,(0,1,2,3))))
        if self.pbias is not None:
            param = torch.cat((self.pbias,self.nbias),0)
            reg += torch.sum(torch.sqrt(1e-8+torch.sum(param**2,0)))
        return reg
         
    
        
        
        
        
        
