#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# -------------------------------------------------------
# General tools
# Ref: Bayesian Compression for Deep Learning
# -------------------------------------------------------


def unit_round_off(t=23):
    """
    :param t:
        number significand bits
    :return:
        unit round off based on nearest interpolation, for reference see [1]
    """
    return 0.5 * 2. ** (1. - t)


SIGNIFICANT_BIT_PRECISION = [unit_round_off(t=i + 1) for i in range(23)]


def float_precision(x):

    out = np.sum([x < sbp for sbp in SIGNIFICANT_BIT_PRECISION])
    return out


def float_precisions(X, dist_fun, layer=1):

    X = X.flatten()
    out = [float_precision(2 * x) for x in X]
    out = np.ceil(dist_fun(out))
    return out


def special_round(input, significant_bit):
    delta = unit_round_off(t=significant_bit)
    rounded = np.floor(input / delta + 0.5)
    rounded = rounded * delta
    return rounded


def fast_infernce_weights(w, exponent_bit, significant_bit):

    return special_round(w, significant_bit)


def compress_matrix(x):

    if len(x.shape) != 2:
        A, B, C, D = x.shape
        x = x.reshape(A * B,  C * D)
        # remove non-necessary filters and rows
        x = x[:, (x != 0).any(axis=0)]
        x = x[(x != 0).any(axis=1), :]
    else:
        # remove unnecessary rows, columns
        x = x[(x != 0).any(axis=1), :]
        x = x[:, (x != 0).any(axis=0)]
    return x

def check(x,axis):
   a=  (x < 0.02).any(axis=axis)
   
   return a
def compress_matrix_(x):
  
    if len(x.shape) != 2:
        A, B, C, D = x.shape
        x = x.reshape(A * B,  C * D)
        # remove non-necessary filters and rows
        x = x[:, check(x,0)]
        x = x[check(x,1), :]
    else:
        # remove unnecessary rows, columns
        x = x[check(x,1), :]
        x = x[:, check(x,0)]
    return x

def compute_posterior_params(layer):
    return layer.posterior_mean,F.softplus(layer.posterior_un_scale,beta=10)

def extract_pruned_params(layers, masks):

    post_weight_mus = []
    post_weight_vars = []

    for i, (layer, mask) in enumerate(zip(layers, layers)):
        # compute posteriors
        post_weight_mu, post_weight_var = compute_posterior_params(layer)
        post_weight_var = post_weight_var.cpu().data.numpy()
        post_weight_mu  = post_weight_mu.cpu().data.numpy()
        # apply mask to mus and variances
        post_weight_mu  = post_weight_mu #* mask
        post_weight_var = post_weight_var #* mask

        post_weight_mus.append(post_weight_mu)
        post_weight_vars.append(post_weight_var)

    return post_weight_mus, post_weight_vars

def mycompression(m,v):
    print(m.shape)
    full=np.where(np.abs(m)<-10,0,1)
    mask=np.where(np.abs(m)/v<0.35,0,1)
    print(np.sum(mask)/np.sum(full))
    return mask

# -------------------------------------------------------
#  Compression rates (fast inference scenario)
# -------------------------------------------------------


def _compute_compression_rate(vars, in_precision=32., dist_fun=lambda x: np.max(x), overflow=10e38,underflow=0):
    # compute in  number of bits occupied by the original architecture
    sizes      = [v.size for v in vars]
    nb_weights = float(np.sum(sizes))
    IN_BITS    = in_precision * nb_weights
    # prune architecture
    
    vars = [compress_matrix(v) for v in vars]
    sizes = [v.size for v in vars]
   
    
    # compute
    significant_bits = [float_precisions(v, dist_fun, layer=k + 1) for k, v in enumerate(vars)]
  
    exponet_const=-np.ceil(np.log2(underflow))
    exponet_const=2**np.ceil(np.log2(exponet_const))-1
    exponet_const=min(exponet_const,127)
    exponent_bit = np.ceil(np.log2(np.log2(overflow)+exponet_const ) )
    
   
    total_bits = [1. + exponent_bit + sb for sb in significant_bits]
    OUT_BITS = np.sum(np.asarray(sizes) * np.asarray(total_bits))
   
   
    return nb_weights / np.sum(sizes), IN_BITS / OUT_BITS, significant_bits, exponent_bit


def rm(weight_mus, weight_vars ):
    for i in range(len(weight_mus)):
        
        mask=mycompression(weight_mus[i], weight_vars[i])
        weight_mus[i], weight_vars[i]=weight_mus[i]*mask, weight_vars[i]*mask
    return  weight_mus, weight_vars 

def compute_compression_rate(layers, masks):
    # reduce architecture
    weight_mus, weight_vars = extract_pruned_params(layers, masks)

    # compute overflow level based on maximum weight
  
    overflow  = np.max([np.max(np.abs(w)) for w in weight_mus])
    underflow = np.min([np.min(np.abs(w)) for w in weight_vars])
    # compute compression rate
    CR_architecture, CR_fast_inference, _, _ = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow,underflow=underflow)
   
    print("Making use of weight uncertainty can reduce the model by a factor of %.1f." % (CR_fast_inference))
    print("Memory Reduction ",np.round((100-100/CR_fast_inference),1),"%"  )
    print("Average needed bits %.1f" % (32/CR_fast_inference))
    


def compute_reduced_weights(layers, masks):
    weight_mus, weight_vars = extract_pruned_params(layers, masks)
    overflow  = np.max([np.max(np.abs(w)) for w in weight_mus])
    underflow = np.min([np.min(np.abs(w)) for w in weight_vars])
    _, _, significant_bits, exponent_bits = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow,underflow=underflow)
    weights = [fast_infernce_weights(weight_mu, exponent_bits, significant_bit) for weight_mu, significant_bit in
               zip(weight_mus, significant_bits)]
    return weights
