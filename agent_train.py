import numpy as np
import random
from random import randrange
import torch
import torch.nn.functional as F
from Qnetwork import Qnet,Qnetn,DuelQnet,NoisyQnet
from Experience_replay_buffer import Replay_buffer,Prioritized_replay_buffer
import torch.autograd as autograd 
import torch.optim as optim 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_shape, action_shape, seed, memory_size, gamma, lr,update_frequency,beta=0,norm=False,double=False,duel=False,noisy=False):
        """Initialize an Agent.
        
        Params
        ======
            state_shape (int): dimension of each state
            action_shape (int): dimension of each action
            seed (int): random seed
            memory_size(int): replay buffer size
            gamma(float):discount factor for Q-learning algorithm
            lr(float):learning rate
            update_frequency(int): how often to update target network
            beta(float):beta for prioritized replay buffer,if beta=0, the use normal replay buffer
            norm(boolean):select Q-network with normalization layers as agent's NN
            duel(boolean):select duel Q-network as agent's NN
            noisy(boolean):select noisy Q-network as agent's NN
            double(boolean):select double target value for NN update
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.seed = random.seed(seed)
        self.gamma=gamma
        self.update=update_frequency
        self.steps = 0
        self.double=double
        self.noisy=noisy
        self.duel=duel
        self.pr=[]
        # Q-Network
        if self.noisy==True:
            self.net = NoisyQnet(state_shape, action_shape, seed).to(device)
            self.t_net = NoisyQnet(state_shape, action_shape, seed).to(device)
        else:
            if self.duel==True:
                self.net = DuelQnet(state_shape, action_shape, seed).to(device)
                self.t_net = DuelQnet(state_shape, action_shape, seed).to(device)
            else:
                if norm==True:
                  self.net = Qnetn(state_shape, action_shape, seed).to(device)
                  self.t_net = Qnetn(state_shape, action_shape, seed).to(device)
                else:
                  self.net = Qnet(state_shape, action_shape, seed).to(device)
                  self.t_net = Qnet(state_shape, action_shape, seed).to(device)
        self.t_net.load_state_dict(self.net.state_dict())
        self.t_net.eval()
        #different optimizer
        #self.optimizer = optim.RMSprop(self.net.parameters(),lr=LR, eps=0.01, alpha=0.95,momentum=0.95)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.beta = beta
        # Replay memory
        if(self.beta==0):
            self.memory = replay_buffer(memory_size)
        else:
            self.memory = Prioritized_replay_buffer(memory_size)

    
    def step(self, state, action, reward, next_state, done,batch_size,min_frames):
        """Execute a learning step for agent's current transaction
        Params
        ======
            state(torch.Tensor):environment's state
            action(int):agent's action
            reward(float):environment's reward
            done(boolean):environment's final state
            batch_size(int):batch size for updating NN's weights
            min_frames(int):minimum number of steps before start learning
        """
        #   Save experience in  memory
        self.memory.add_data(state, action, reward, next_state, done)
        #   Learn with 'update' frequency of steps
        self.steps = (self.steps+1)%self.update
        if self.steps==0 and len(self.memory)>min_frames:
            return self.opt_model(batch_size,self.beta,self.double)
        else:
            return 0

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

    def opt_model(self,batch_size,beta,double=False):
        """Update NN' weights according to DQN or double DQN algorithm 
        Params
        ======
            batch_size(int):batch size for updating NN's weights
            beta(float):beta for prioritized update value parameters
            double(boolean):for double target value
        """
        if len(self.memory)<batch_size:
            return
        if beta==0:
            oldstate,action,reward,state,done=self.memory.sampling(batch_size)
        else:
            oldstate,action,reward,state,done,idxes,weights=self.memory.sampling(batch_size,beta)
            weights =torch.tensor(weights.reshape(-1,1)).to(device)
        oldstate = torch.tensor(np.array(oldstate/255.0, copy=False)).to(device)
        action= torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        done_mask = torch.BoolTensor(done).to(device)
        Q_net_vals = self.net(oldstate.float()).gather(1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            state = torch.tensor(np.array(state/255.0, copy=False)).to(device)
            if double:
                Q_t_net_acts = self.net(state.float()).max(1)[1].unsqueeze(-1)
                Q_t_net_vals = self.t_net(state.float()).gather(1,Q_t_net_acts).squeeze(-1)
            else:
                Q_t_net_vals = self.t_net(state.float()).max(1)[0]
            Q_t_net_vals[done_mask] = 0.0
            Q_t_net_vals=Q_t_net_vals.detach()
            y = reward + self.gamma  * Q_t_net_vals 

        if beta==0:
            #Huber loss
            loss = F.smooth_l1_loss(Q_net_vals,y.float())
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        else:
            loss_i = F.smooth_l1_loss(Q_net_vals,y.float(), reduction="none")
            loss = torch.mean(loss_i*weights)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            loss_for_prio = loss_i.detach().cpu().numpy()
            prios = loss_for_prio + self.memory.eps
            self.pr=prios
            self.memory.update_priorities(idxes,prios)
        #reset weight's noise
        if self.noisy:
          self.net.reset_noise()
          self.t_net.reset_noise()
          
        return loss.item()
    

                   
