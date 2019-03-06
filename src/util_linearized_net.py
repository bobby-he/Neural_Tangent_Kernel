"""In this file I will define some utilities functions that will be used in the experiments on the linearized approximation of the network."""

# Import stuff
 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from NTK_net import cpu_tuple, circle_transform


# compute the list of gradients of the prediction of the net over the different training points (denoted by gamma_data)
# with respect to all parameters of the network. 
def compute_grad_list(net, gamma_data, use_cuda=False): 
  grad_list = []
  for gamma in gamma_data:
    circle_pt = circle_transform(gamma)
    if use_cuda and torch.cuda.is_available():
      circle_pt = circle_pt.cuda()
    loss = net(circle_pt)
    grad_list.append(cpu_tuple(torch.autograd.grad(loss, net.parameters(), retain_graph = True)))
    
  return(grad_list)


# compute the tangent kernel from the list of gradients, with respect to the input points for which the list of gradients was computed.
def compute_theta_0(grad_list):
  n = len(grad_list)  
  theta_0 = torch.zeros((n,n))
  for i in range(n):
    grad_i = grad_list[i]
    for j in range(i+1):
      grad_j = grad_list[j]
      theta_0[i, j] = sum([torch.sum(torch.mul(grad_i[u], grad_j[u])) for u in range(len(grad_j))])
      theta_0[j, i] = theta_0[i, j]
      
  return(theta_0)


# utility function to unpack the list of gradients in a matrix with shape (n_train_points, n_parameters), essentially stretching all 
# parameters of the network into a single dimension
def unpack_gradients(grad_list):
  final_tensor = torch.tensor([])
  for i in range(len(grad_list)):
    tmp = torch.tensor([]*len(grad_list))
    for j in range(len(grad_list[0])):
      tmp = torch.cat((tmp, grad_list[i][j].reshape(-1)))

    if final_tensor.nelement()==0:
      final_tensor = tmp.clone().detach().reshape(1, -1)
    else: 
      final_tensor = torch.cat((final_tensor, tmp.reshape(1, -1)))
  return(final_tensor)


# same as above but for the weights of the net; you get a vector of shape (n_parameters, )
def unpack_weights(weights_list, use_cuda=False):
  tmp = torch.tensor([]).to("cuda" if use_cuda else "cpu")
  for j in range(len(weights_list)):
    tmp = torch.cat((tmp, weights_list[j].reshape(-1)))

  return(tmp)


# function to obtain a vector with all parameters of the net
def obtain_weights(net): 
  weights = list(net.parameters())
  weights = unpack_weights(weights).detach().cpu().numpy()
  return(weights)