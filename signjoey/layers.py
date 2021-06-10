import torch, math
from torch.nn import Module, Parameter
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from signjoey.utils import concrete_sample, kumaraswamy_sample, bin_concrete_sample, kl_divergence_kumaraswamy
import numpy as np
import weakref
import pandas as pd
import math
import torch
import time
from torch import nn, Tensor
from signjoey.helpers import freeze_params




class DenseBayesian(Module):
    """
    Class for a Bayesian Dense Layer employing various activations, namely ReLU, Linear and LWTA, along with IBP.
    """
    instances=weakref.WeakSet()
    ID=0
    simplified_inference= False
    def __init__(self, input_features, output_features, competitors,
                 activation, deterministic = False, temperature = 0.67, ibp = False, bias=True,prior_mean=1,prior_scale=1,kl_w=1.0,name=None,out_w=False,init_w=1.0,scale_out=1.0):
        """

        :param input_features: int: the number of input_features
        :param output_features: int: the number of output features
        :param competitors: int: the number of competitors in case of LWTA activations, else 1
        :param activation: str: the activation to use. 'relu', 'linear' or 'lwta'
        :param prior_mean: float: the prior mean for the gaussian distribution
        :param prior_scale: float: the prior scale for the gaussian distribution
        :param temperature: float: the temperature of the relaxations.
        :param ibp: boolean: flag to use the IBP prior.
        :param bias: boolean: flag to use bias.
        :param kl_w: float: weight on the KL loss contribution
        :param scale_out: float: scale layers ouputput
        """

        super(DenseBayesian, self).__init__()
        
       
        self.scale_out=scale_out
        self.ID=DenseBayesian.ID
        DenseBayesian.ID+=1
        self.name=name
        self.n=0.0001
        
        self.kl_w=kl_w
        self.init_w=init_w
        DenseBayesian.instances.add(self)
        self.input_features = input_features
        self.output_features = output_features
        self.K = output_features // competitors
        self.U = competitors
        self.activation = activation
        self.deterministic = deterministic

        self.temperature = 1.67#temperature
        self.ibp = ibp
        self.bias  = bias
        self.tau = 1e-2
        self.training = True
        self.out_wYN=out_w
        
        if out_w:
            self.out_w=Parameter(torch.Tensor(1))
      #  self.out_w.to('cuda')
        #################################
        #### DEFINE THE PARAMETERS ######
        #################################

        self.posterior_mean = Parameter(torch.Tensor(output_features, input_features))

        if not deterministic:
            # posterior unnormalized scale. Needs to be passed by softplus
            self.posterior_un_scale = Parameter(torch.Tensor(output_features, input_features))
            self.register_buffer('weight_eps', None)

        if activation == 'lwta':
            if competitors == 1:
                print('Cant perform competition with 1 competitor.. Setting to default: 4\n')
                self.U = 4
                self.K = output_features // 4
            if output_features % self.U != 0:
                raise ValueError('Incorrect number of competitors. '
                                 'Cant divide {} units in groups of {}..'.format(output_features, competitors))
                
     
        else:
            if competitors != 1:
                print('Wrong value of competitors for activation ' + activation + '. Setting value to 1.')
                self.K = output_features
                self.U = 1

        #########################
        #### IBP PARAMETERS #####
        #########################
        if ibp:
            self.prior_conc1 = torch.tensor(1.)
            self.prior_conc0 = torch.tensor(1.)

            self.conc1 = Parameter(torch.Tensor(self.K))
            self.conc0 = Parameter(torch.Tensor(self.K))

            self.t_pi = Parameter(torch.Tensor(input_features, self.K))
        else:
            self.register_parameter('prior_conc1', None)
            self.register_parameter('prior_conc0', None)
            self.register_parameter('conc1', None)
            self.register_parameter('conc0', None)
            self.register_parameter('t_pi', None)

        if bias:
            self.bias_mean = Parameter(torch.Tensor(output_features))

            if not deterministic:
                self.bias_un_scale = Parameter(torch.Tensor(output_features))
                self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mean', None)
            if not deterministic:
                self.register_parameter('bias_un_scale', None)
                self.register_buffer('bias_eps', None)


        self.reset_parameters()
   

    def reset_parameters(self):
        """
        Initialization function for all the parameters of the Dense Bayesian Layer.

        :return: null
        """

        # can change this to uniform with std or something else
        #stdv = 1. / math.sqrt(self.posterior_mean.size(1))
        #self.posterior_mean.data.uniform_(-stdv, stdv)

        # original init
        #init.xavier_normal_(self.posterior_mean)
        init.kaiming_uniform_(self.posterior_mean, a = 0.01*math.sqrt(5))
        if not self.deterministic:
            self.posterior_un_scale.data.fill_(-0.125)

        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.posterior_mean)
            bound = 1. / math.sqrt(fan_in)
            init.uniform_(self.bias_mean, -bound*0.1*self.init_w, bound*0.1)
            #self.bias_mean.data.fill_(0.1)

            if not self.deterministic:
                self.bias_un_scale.data.fill_(-0.5)

        if self.ibp:
            self.conc1.data.fill_(2.)
            self.conc0.data.fill_(.5453)

            init.uniform_(self.t_pi, .1, 1.)
     
    def forward(self, input):
        """
        Override the default forward function to implement the Bayesian layer.

        :param input: torch tensor: the input data

        :return: torch tensor: the output of the current layer
        """
      
        layer_loss = 0.
        self.n=0
        if self.training:

            if not self.deterministic:
                # use the reparameterization trick
                posterior_scale = F.softplus(self.posterior_un_scale,beta=10)
                W = self.posterior_mean + posterior_scale * torch.randn_like(self.posterior_un_scale)
                kl_weights = -0.5 * torch.sum(2*torch.log(posterior_scale) - torch.square(self.posterior_mean)
                                               - torch.square(posterior_scale) + 1)
                layer_loss += torch.sum(kl_weights)
                self.n += len(self.posterior_mean.view(-1))

            else:
              
                W = self.posterior_mean


            if self.ibp:
                z, kl_sticks, kl_z = self.indian_buffet_process(self.temperature)

                W = z.T*W

                layer_loss += kl_sticks
                layer_loss += kl_z

            if self.bias:
                if not self.deterministic:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale,beta=10) * torch.randn_like(self.bias_un_scale)
                    bias_kl = -0.5 * torch.sum(2*torch.log(F.softplus(self.bias_un_scale,beta=10)) - 
                                                   torch.square(self.bias_mean)
                                                   - torch.square(F.softplus(self.bias_un_scale,beta=10)) + 1)
                    self.n += len(self.bias_mean.view(-1))
                    layer_loss += torch.sum(bias_kl)
                else:
                    bias = self.bias_mean
            else:
                bias = None

        else:
            
            if DenseBayesian.simplified_inference or self.deterministic:
                W = self.posterior_mean
            else:
                posterior_scale = F.softplus(self.posterior_un_scale,beta=10)
                W = self.posterior_mean + posterior_scale * torch.randn_like(self.posterior_un_scale)

            if self.bias:
                if DenseBayesian.simplified_inference or self.deterministic :
               
                    bias = self.bias_mean
                else:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale,beta=10) * torch.randn_like(self.bias_un_scale)
            else:
                bias = None

            if self.ibp:
                z, _, _ = self.indian_buffet_process(0.01)
                W = z.T*W

        out = F.linear(input, W, bias)
        if self.out_wYN:
            out=out*torch.sigmoid(self.out_w).to('cuda')
            layer_loss=layer_loss+torch.sigmoid(self.out_w)
            if np.random.uniform()<0.001:
                print(torch.sigmoid(self.out_w))
    
        if self.activation == 'linear':
            self.loss = layer_loss
            self.loss*=self.kl_w
            
            return out*self.scale_out

        elif self.activation == 'relu':
            self.loss = layer_loss
            self.loss*=self.kl_w
            return F.relu(out)*self.scale_out

        elif self.activation == 'lwta':
            out, kl =  self.lwta_activation(out, self.temperature if self.training else 0.01)
            layer_loss += kl
            self.loss = layer_loss
            self.loss*=self.kl_w
            return out*self.scale_out
       
        else:
            raise ValueError(self.activation + " is not implemented..")


    def indian_buffet_process(self, temp = 0.67):

        kl_sticks = kl_z = 0.
        z_sample = bin_concrete_sample(self.t_pi, temp)

        if not self.training:
            t_pi_sigmoid = torch.sigmoid(self.t_pi)
            mask = t_pi_sigmoid >self.tau
            z_sample = t_pi_sigmoid*mask

        z = z_sample.repeat(1, self.U)

        # compute the KL terms
        if self.training:

            a_soft = F.softplus(self.conc1)
            b_soft = F.softplus(self.conc0)

            q_u = kumaraswamy_sample(a_soft, b_soft, sample_shape = [self.t_pi.size(0), self.t_pi.size(1)])
            prior_pi = torch.cumprod(q_u, -1)

            q = torch.sigmoid(self.t_pi)
            log_q = torch.log(q + 1e-6)
            log_p = torch.log(prior_pi + 1e-6)
            
            kl_z = torch.sum(q*(log_q - log_p))
            kl_sticks = torch.sum(kl_divergence_kumaraswamy(torch.ones_like(a_soft), a_soft, b_soft))
            self.n += len(self.t_pi.view(-1))
            self.n += len(a_soft.view(-1))

        return z, kl_sticks, kl_z


    def lwta_activation(self, input, temp = 0.67, hard = False):
        """
        Function implementing the LWTA activation with a competitive random sampling procedure as described in
        Panousis et al., Nonparametric Bayesian Deep Networks with Local Competition.

        :param hard: boolean: flag to draw hard samples from the binary relaxations.
        :param input: torch tensor: the input to compute the LWTA activations.

        :return: torch tensor: the output after the lwta activation
        """

        kl = 0.

        logits = torch.reshape(input, [-1,input.size(-2), self.K, self.U])
       
            
        xi = concrete_sample(logits, temperature = temp, hard = hard,rand=True)
        out = logits*xi
      
        out = out.reshape(input.shape)
    
        if self.training:
            q = F.softmax(logits, -1)
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0/self.U))

            kl = torch.sum(q*(log_q - log_p),1)
            kl = torch.sum(kl)
            self.n+=len(q.view(-1))
            #scale up to be comparable with other terms
            kl=kl*100
        return out, kl

    
    
  
        
        
    def extra_repr(self):
        """
        Print some stuff about the layer parameters.

        :return: str: string of parameters of the layer.
        """

        return "input_features = {}, output_features = {}, bias = {}".format(
            self.input_features, self.output_features, self.bias
        )



        return "prior_mean = {}, prior_scale = {}, input_features = {}, output_features = {}, bias = {}".format(
            self.prior_mean, self.prior_scale, self.in_channels, self.out_channels, self.bias
        )

    
    
    
    

class EmbeddingBayesian(DenseBayesian,Module):

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    max_norm: float
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None,
                 max_norm: float = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight = None, _vars = None,input_features=0,output_features=0, competitors=1,
                 activation='relu',*args,**kwargs) -> None:
        super(Module).__init__()
        super(EmbeddingBayesian,self).__init__(input_features, output_features, competitors,activation)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.input_features =num_embeddings
        self.output_features=embedding_dim
        self.competitors=4
        self.activation='lwta'
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

        self.sparse = sparse
        print(self.ID)

        time.sleep(10)


    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        """
        Override the default forward function to implement the Bayesian layer.

        :param input: torch tensor: the input data

        :return: torch tensor: the output of the current layer
        """
      
        layer_loss = 0.
        self.n=0
        if self.training:

            if not self.deterministic:
                # use the reparameterization trick
                posterior_scale = F.softplus(self.posterior_un_scale,beta=10)
                W = self.posterior_mean + posterior_scale * torch.randn_like(self.posterior_un_scale)
                kl_weights = -0.5 * torch.sum(2*torch.log(posterior_scale) - torch.square(self.posterior_mean)
                                               - torch.square(posterior_scale) + 1)
                layer_loss += torch.sum(kl_weights)
                self.n += len(self.posterior_mean.view(-1))

            else:
              
                W = self.posterior_mean


            if self.ibp:
                z, kl_sticks, kl_z = self.indian_buffet_process(self.temperature)

                W = z.T*W

                layer_loss += kl_sticks
                layer_loss += kl_z

            if self.bias:
                if not self.deterministic:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale,beta=10) * torch.randn_like(self.bias_un_scale)
                    bias_kl = -0.5 * torch.sum(2*torch.log(F.softplus(self.bias_un_scale,beta=10)) - 
                                                   torch.square(self.bias_mean)
                                                   - torch.square(F.softplus(self.bias_un_scale,beta=10)) + 1)
                    self.n += len(self.bias_mean.view(-1))
                    layer_loss += torch.sum(bias_kl)
                else:
                    bias = self.bias_mean
            else:
                bias = None

        else:
            
            if DenseBayesian.simplified_inference or self.deterministic:
                W = self.posterior_mean
            else:
                posterior_scale = F.softplus(self.posterior_un_scale,beta=10)
                W = self.posterior_mean + posterior_scale * torch.randn_like(self.posterior_un_scale)

            if self.bias:
                if DenseBayesian.simplified_inference or self.deterministic :
               
                    bias = self.bias_mean
                else:
                    bias = self.bias_mean + F.softplus(self.bias_un_scale,beta=10) * torch.randn_like(self.bias_un_scale)
            else:
                bias = None

            if self.ibp:
                z, _, _ = self.indian_buffet_process(0.01)
                W = z.T*W
      
        out = F.embedding(
            input, W.T, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
       
        if self.out_wYN:
            out=out*torch.sigmoid(self.out_w).to('cuda')
            layer_loss=layer_loss+torch.sigmoid(self.out_w)
            if np.random.uniform()<0.001:
                print(torch.sigmoid(self.out_w))
    
        if self.activation == 'linear':
            self.loss = layer_loss
            self.loss*=self.kl_w
            
            return out*self.scale_out

        elif self.activation == 'relu':
            self.loss = layer_loss
            self.loss*=self.kl_w
            return F.relu(out)*self.scale_out

        elif self.activation == 'lwta':
            out, kl =  self.lwta_activation(out, self.temperature if self.training else 0.01)
            layer_loss += kl
            self.loss = layer_loss
            self.loss*=self.kl_w
            return out*self.scale_out
       
        else:
            raise ValueError(self.activation + " is not implemented..")
         

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
       
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding



