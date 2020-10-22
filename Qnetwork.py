import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Qnet(nn.Module):
    """Q network for Q value estimation"""
    def __init__(self, input_shape, action_shape,seed=0):
        """Initialize random seed and build model.
        Params
        ======
            input_shape (int): Shape of input
            action_shape (int): Shape of action
            seed (int): Random seed
        The commented lines are uded to calculate the flatten conv3 exit
        to determine fc1 
        """
        super(Qnet, self).__init__()
        
        #size=84
        self.seed = torch.manual_seed(seed)
        kernel_size_1=8
        stride_1=4
        self.conv1 = nn.Conv2d(input_shape[0],32, 8, 4)
        #size=(size - (kernel_size_1 - 1) - 1) // stride_1  + 1
        kernel_size_2=4
        stride_2=2
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        #size=(size - (kernel_size_2 - 1) - 1) // stride_2  + 1
        kernel_size_3=3
        stride_3=1
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        #size=(size - (kernel_size_2 - 1) - 1) // stride_2  + 1
        #convw=size
        #convh=size
        #linear_input_size = convw * convw * 64
        #print(linear_input_size,convw,convh)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_shape)

        
    def forward(self, x):
        """Build the neural network that maps state to action values."""
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

class Qnetn(nn.Module):
    """Q network with normalization layers for Q value estimation"""
        """Initialize random seed and build model.
        Params
        ======
            input_shape (int): Shape of input
            action_shape (int): Shape of action
            seed (int): Random seed
        """
    def __init__(self, input_shape, action_shape,seed=0):
        super(Qnetn, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(input_shape[0],32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_shape)

        
    def forward(self, x):
        """Build the neural network that maps state to action values."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

class DuelQnet(nn.Module):
    """Duel Q network for Q value estimation"""
    def __init__(self, input_shape, action_shape,seed=0):
        """Initialize random seed and build model.
        Params
        ======
            input_shape (int): Shape of input
            action_shape (int): Shape of action
            seed (int): Random seed
        """
        
        super(DuelQnet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(input_shape[0],32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fca1 = nn.Linear(64*7*7, 512)
        self.fca2 = nn.Linear(512, action_shape)
        self.fcv1 = nn.Linear(64*7*7, 512)
        self.fcv2 = nn.Linear(512, 1)
        
        
    def forward(self, x):
        """Build the neural network that maps state to action values."""
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = x.view(x.size(0), -1)
        advantage = self.fca2(F.relu(self.fca1(x)))
        value = self.fcv2(F.relu(self.fcv1(x)))
        return value + advantage - advantage.mean()



class NoisyLinear(nn.Module):
    """Noisy layer"""
    def __init__(self, in_features, out_features, std_init=0.5):
        """Initialize random seed and build model.
        Params
        ======
            input_shape (int): Shape of input
            action_shape (int): Shape of action
            seed (int): Random seed
            std_init(float):standard deviation for Gaussian noise
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def forward(self, x):
        """add noise to layer's weights and bias"""
        if self.training: 
            weight = self.weight_mu + self.weight_sigma*self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma*self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        """reset layer's weights and bias"""
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        """reset layer's noise"""
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        """scale layer's noise"""
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
class NoisyQnet(nn.Module):
    """Noisy Q network for Q value estimation"""
    def __init__(self, input_shape, action_shape,seed=0):
        """Initialize random seed and build model.
        Params
        ======
            input_shape (int): Shape of input
            action_shape (int): Shape of action
            seed (int): Random seed
            std_init(float):standard deviation for Gaussian noise
        """
        super(NoisyQnet, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(input_shape[0],32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.noisy1=NoisyLinear(64*7*7, 512)
        self.noisy2=NoisyLinear(512, action_shape)


    def forward(self, x):
        """Build the neural network that maps state to action values"""
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu(self.noisy1(x.view(x.size(0), -1)))
        return self.noisy2(x)
    
    def reset_noise(self):
        """Reset noise into neural network's layers"""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()



