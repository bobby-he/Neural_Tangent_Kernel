# Import stuff
 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearNeuralTangentKernel(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=0.1, w_sig = 1):
        self.beta = beta
        super(LinearNeuralTangentKernel, self).__init__(in_features, out_features)
        self.reset_parameters()
        self.w_sig = w_sig
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        return F.linear(input, self.w_sig * self.weight/np.sqrt(self.in_features), self.beta * self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta
        )

class FourLayersNet(nn.Module):

    def __init__(self, n_wid, n_out = 1, beta=0.1):
        super(FourLayersNet, self).__init__()
        self.fc1 = LinearNeuralTangentKernel(2, n_wid, beta=beta)
        self.fc2 = LinearNeuralTangentKernel(n_wid, n_wid, beta=beta)
        self.fc3 = LinearNeuralTangentKernel(n_wid, n_wid, beta=beta)
        self.fc4 = LinearNeuralTangentKernel(n_wid, n_out, beta=beta)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def circle_transform(angle_vec):
    cos_tensor = torch.cos(angle_vec)
    sin_tensor = torch.sin(angle_vec)
    return torch.stack((cos_tensor, sin_tensor), -1).float()

def train_net(net, n_epochs, input_data, target, lr=1):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(n_epochs):  
        optimizer.zero_grad()


        outputs = net(input_data)

        loss = criterion(outputs.view(-1), target)/2

        loss.backward()

        optimizer.step()
  
def variance_est(n_width, n_pts, temp_mat, n_nets):
    with torch.no_grad():    
        gamma_test = torch.tensor(np.linspace(-np.pi,np.pi, n_pts))
        gamma_data = torch.tensor(np.array([-2.2, -1, 1, 2.2]))
        input_data = circle_transform(gamma_data)
        circle_test = circle_transform(gamma_test)
        net = FourLayersNet(n_width, n_out = n_nets).cuda()
        sig_testvtest = torch.var(net(circle_test.cuda()), dim = 1, keepdim = True).cpu()
        sig_testvtrain = torch.mm(net(circle_test.cuda()), torch.t(net(input_data.cuda()))).cpu()/n_nets
        sig_trainvtrain =torch.mm(net(input_data.cuda()), torch.t(net(input_data.cuda()))).cpu()/n_nets
        variance_vec = sig_testvtest.view(-1) -2 * torch.diag(torch.mm(sig_testvtrain,torch.t(temp_mat))) + torch.diag(torch.mm(temp_mat, torch.mm(sig_trainvtrain, torch.t(temp_mat))))
        variance_vec = variance_vec.cpu()
        return np.maximum(variance_vec, 0)

# saves gradient objects onto cpu, saves GPU memory
def cpu_tuple(tuple_obj):
    return tuple([obj.cpu() for obj in tuple_obj])

def kernel_leastsq_update(test_output, train_output, K_testvtrain, K_trainvtrain, train_target, n_steps = 1): 
    test_output = test_output + n_steps/len(train_target) * np.matmul(K_testvtrain, train_target - train_output).flatten()
    train_output = train_output + n_steps/len(train_target) * np.matmul(K_trainvtrain, train_target - train_output).flatten()
    return test_output, train_output


class AnimationPlot_lsq(object):
    def __init__(self, n_nets, input_data, K_testvtrain, K_trainvtrain, train_target, line_tuple, ax,
       n_wid = 50, n_out = 1, n_pts = 100, epochs_per_frame = 1):
        self.line1, self.line2, self.line3, self.line4, self.line0, self.line1a,\
        self.line2a, self.line3a, self.line4a, self.line0a = line_tuple
        self.test_output = np.zeros((n_pts, n_nets))
        self.train_output = np.zeros((4, n_nets))
        self.epochs_per_frame = epochs_per_frame
        self.n_nets = n_nets
        self.input_data = input_data
        self.train_target = train_target
        self.K_testvtrain = K_testvtrain
        self.K_trainvtrain = K_trainvtrain
        self.train_target = train_target
        self.gamma_vec = torch.tensor(np.linspace(-np.pi, np.pi, n_pts))
        self.circle_test = circle_transform(self.gamma_vec).cuda()
        self.ax = ax

        for i in range(self.n_nets):
            self.__dict__['net {}'.format(i)] = FourLayersNet(n_wid, n_out).cuda()
            self.test_output[:, i] = self.__dict__['net {}'.format(i)](self.circle_test).cpu().detach().numpy().flatten()
            self.train_output[:, i] = self.__dict__['net {}'.format(i)](self.input_data.cuda()).cpu().detach().numpy().flatten()

    def step(self):
        for i in range(self.n_nets):
            train_net(self.__dict__['net {}'.format(i)], self.epochs_per_frame, self.input_data, self.train_target)
            self.test_output[:, i], self.train_output[:, i] = kernel_leastsq_update(self.test_output[:, i],  self.train_output[:, i],
                                                                              self.K_testvtrain, self.K_trainvtrain,
                                                                              self.train_target.cpu().detach().numpy())
      
    def plot_step(self, i):
        j = 0
        if i>2:
            self.step()
            j = i - 2

        self.line0.set_data(self.gamma_vec.numpy(), self.__dict__['net {}'.format(0)](self.circle_test).cpu().detach().numpy())
        self.line0a.set_data(self.gamma_vec.numpy(), self.test_output[:, 0])
        if self.n_nets > 1:
            self.line1.set_data(self.gamma_vec.numpy(), self.__dict__['net {}'.format(1)](self.circle_test).cpu().detach().numpy())
            self.line1a.set_data(self.gamma_vec.numpy(), self.test_output[:, 1])
        if self.n_nets > 2:
            self.line2a.set_data(self.gamma_vec.numpy(), self.test_output[:, 2])
            self.line2.set_data(self.gamma_vec.numpy(), self.__dict__['net {}'.format(2)](self.circle_test).cpu().detach().numpy())

        if self.n_nets > 3:
            self.line3.set_data(self.gamma_vec.numpy(), self.__dict__['net {}'.format(3)](self.circle_test).cpu().detach().numpy())
            self.line3a.set_data(self.gamma_vec.numpy(), self.test_output[:, 3])
        if self.n_nets >4:
            self.line4.set_data(self.gamma_vec.numpy(), self.__dict__['net {}'.format(4)](self.circle_test).cpu().detach().numpy())
            self.line4a.set_data(self.gamma_vec.numpy(), self.test_output[:, 4])

        self.ax.set_title('Epoch {}'.format(j))

        if self.n_nets ==1:
            return(self.line0, self.line0a,)
        if self.n_nets ==2:
            return(self.line0, self.line0a, self.line1, self.line1a,)
        if self.n_nets ==3:
            return(self.line0, self.line0a, self.line1, self.line1a, self.line2, self.line2a,)
        if self.n_nets ==4:
            return(self.line0, self.line0a, self.line1, self.line1a, self.line2, self.line2a, self.line3, self.line3a,)
        if self.n_nets ==5:
            return (self.line1, self.line2, self.line3, self.line4, self.line0, self.line1a, self.line2a, self.line3a, self.line4a, self.line0a,)
        
      
def kernel_mats(net, gamma_train, gamma_test, use_cuda = True, n_train = 4, kernels='both'):
  # for a given net, this function computes the K_testvtrain (n_test by n_train) and the
  # K_trainvtrain (n_train by n_train) kernels. 
  # You can choose which one to return by the parameter 'kernels', with values 'both' (default), 'testvtrain' or 'trainvtrain'

  # suppose cuda available
    n_pts = len(gamma_test)
    if use_cuda:
        net = net.cuda()
# the following computes the gradients with respect to all parameters
    grad_list = []
    for gamma in gamma_train:
        circle_pt = circle_transform(gamma)
        if use_cuda:
            circle_pt = circle_pt.cuda()
        loss = net(circle_pt)
        grad_list.append(torch.autograd.grad(loss,net.parameters(), retain_graph = True))
  
# testvstrain kernel
    if kernels=='both' or kernels=='testvtrain':
        K_testvtrain = torch.zeros((n_pts,n_train))
        for i, gamma in enumerate(gamma_test):
            #if ((i+1)*10)%len(gamma_test) == 0:
                #print('K_testvtrain is {}% complete'.format(int((i+1)/len(gamma_test)*100)))
            circle_pt = circle_transform(gamma)
            if use_cuda:
                circle_pt = circle_pt.cuda()
            loss = net(circle_pt)
            grads = torch.autograd.grad(loss,net.parameters(), retain_graph = True) # extract NN gradients 
            for j in range(len(grad_list)):
                pt_grad = grad_list[j] # the gradients at the jth (out of 4) data point
                K_testvtrain[i, j] = sum([torch.sum(torch.mul(grads[u], pt_grad[u])) for u in range(len(grads))])
  
# trainvstrain kernel
    if kernels=='both' or kernels=='trainvtrain':
        K_trainvtrain = torch.zeros((n_train,n_train))
        for i in range(n_train):
            grad_i = grad_list[i]
            for j in range(i+1):
                grad_j = grad_list[j]
                K_trainvtrain[i, j] = sum([torch.sum(torch.mul(grad_i[u], grad_j[u])) for u in range(len(grad_j))])
                K_trainvtrain[j, i] = K_trainvtrain[i, j]
  
    if kernels=='both':
        return K_testvtrain, K_trainvtrain
    elif kernels=='trainvtrain': 
        return K_trainvtrain
    elif kernels=='testvtrain': 
        return K_testvtrain

def kernel_mats_d_gan(net, d_train, d_test, use_cuda = True, kernels='both'):
  # for a given net, this function computes the K_testvtrain (n_test by n_train) and the
  # K_trainvtrain (n_train by n_train) kernels. 
  # You can choose which one to return by the parameter 'kernels', with values 'both' (default), 'testvtrain' or 'trainvtrain'

  # suppose cuda available
    n_test_pts = len(d_test)
    n_train_pts = len(d_train)
    if use_cuda:
        net = net.cuda()
# the following computes the gradients with respect to all parameters
    grad_list = []
    for d_point in d_train:
        if use_cuda:
            d_point = d_point.cuda()
        loss = net(d_point)
        grad_list.append(torch.autograd.grad(loss,net.parameters(), retain_graph = True))
  
# testvstrain kernel
    if kernels=='both' or kernels=='testvtrain':
        K_testvtrain = torch.zeros((n_test_pts,n_train_pts))
        for i, d_point in enumerate(d_test):
            #if ((i+1)*10)%len(d_test) == 0:
                #print('K_testvtrain is {}% complete'.format(int((i+1)/len(d_test)*100)))
            if use_cuda:
                d_point = d_point.cuda()
            loss = net(d_point)
            grads = torch.autograd.grad(loss,net.parameters(), retain_graph = True) # extract NN gradients 
            for j in range(len(grad_list)):
                pt_grad = grad_list[j] # the gradients at the jth (out of 4) data point
                K_testvtrain[i, j] = sum([torch.sum(torch.mul(grads[u], pt_grad[u])) for u in range(len(grads))])
  
# trainvstrain kernel
    if kernels=='both' or kernels=='trainvtrain':
        K_trainvtrain = torch.zeros((n_train_pts,n_train_pts))
        for i in range(len(grad_list)):
            grad_i = grad_list[i]
            for j in range(i+1):
                grad_j = grad_list[j]
                K_trainvtrain[i, j] = sum([torch.sum(torch.mul(grad_i[u], grad_j[u])) for u in range(len(grad_j))])
                K_trainvtrain[j, i] = K_trainvtrain[i, j]
  
    if kernels=='both':
        return K_testvtrain, K_trainvtrain
    elif kernels=='trainvtrain': 
        return K_trainvtrain
    elif kernels=='testvtrain': 
        return K_testvtrain
