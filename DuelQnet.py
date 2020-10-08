import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelQNet(nn.Module):
    def __init__(self, input_shape, action_shape,seed=0):
        super(DuelQNet, self).__init__()
        self.seed = torch.manual_seed(seed)
 
        self.conv1 = nn.Conv2d(input_shape[0],32, 8, 4)

        self.conv2 = nn.Conv2d(32, 64, 4, 2)

        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fca1 = nn.Linear(64*7*7, 512)
        self.fca2 = nn.Linear(512, action_shape)
        self.fcv1 = nn.Linear(64*7*7, 512)
        self.fcv2 = nn.Linear(512, 1)
        
        
    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = x.view(x.size(0), -1)
        advantage = self.fca2(F.relu(self.fca1(x)))
        value = self.fcv2(F.relu(self.fcv1(x)))
        return value + advantage - advantage.mean()


    
