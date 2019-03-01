# Import stuff
 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearNeuralTangentKernel(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=0.1):
      self.beta = beta
      super(LinearNeuralTangentKernel, self).__init__(in_features, out_features)
      self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
          torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        return F.linear(input, self.weight/np.sqrt(self.in_features), self.beta * self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta
        )

class FourLayersNet(nn.Module):

  def __init__(self, n, beta=0.1):
      super(FourLayersNet, self).__init__()
      self.fc1 = LinearNeuralTangentKernel(2, n, beta=beta)
      self.fc2 = LinearNeuralTangentKernel(n, n, beta=beta)
      self.fc3 = LinearNeuralTangentKernel(n, n, beta=beta)
      self.fc4 = LinearNeuralTangentKernel(n, 1, beta=beta)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

def circle_transform(angle_vec):
	cos_tensor = torch.cos(angle_
	sin_tensor = torch.sin(angle_
	return torch.stack((cos_tensor, sin_tensor), -1).float()

def train_net(net, n_epochs, input_data, target):
	criterion = nn.MSELoss(reduction='mean')
	optimizer = optim.SGD(net.parameters(), lr=1)

	for epoch in range(n_epochs):  
		optimizer.zero_grad()

	
		outputs = net(input_data)

		loss = criterion(outputs.view(-1), target)

		loss.backward()

		optimizer.step()
  
  




