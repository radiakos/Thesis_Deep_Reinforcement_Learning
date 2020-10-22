import numpy as np
import random
from random import randrange
import torch
import torch.nn.functional as F
from Qnetwork import Qnet,Qnetn,DuelQnet,NoisyQnet
import torch.autograd as autograd 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_shape, action_shape, norm=False,duel=False,noisy=False):
        """Initialize an Agent object.
        Params
        ======
            state_shape (int): dimension of each state
            action_shape (int): dimension of each action
            norm(boolean):select Q-network with normalization layers as agent's NN
            duel(boolean):select duel Q-network as agent's NN
            noisy(boolean):select noisy Q-network as agent's NN
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        # Q-Network
        if noisy==True:
            self.net = NoisyQnet(state_shape, action_shape).to(device)
        else:
            if duel==True:
                self.net = DuelQnet(state_shape, action_shape).to(device)
            else:
                if norm==True:
                  self.net = Qnetn(state_shape, action_shape).to(device)
                else:
                  self.net = Qnet(state_shape, action_shape).to(device)
        
    def select_action(self,state,eps):
        """Returns actions for given state according to current policy
        Params
        ======
            state (torch.Tensor): current state
            eps (float): epsilon, for eps-greedy action selection
        """
        if random.random()>eps:
            with torch.no_grad():
                t_state=torch.tensor(state/255.0).unsqueeze(0).to(device)
                self.net.eval() 
                state_action_values = self.net(t_state.float())
                self.net.train()
                return int(state_action_values.max(1)[1].view(1, 1))
        else:
            return randrange(self.action_shape)



    
