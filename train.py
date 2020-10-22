import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import os.path as osp
from atari_wrappers import wrap_deepmind
from agent_train import Agent
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""Create environment according to deepmind's preprocessing"""

env = gym.make('MsPacmanNoFrameskip-v0')
env = wrap_deepmind(env)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
state_shape=env.observation_space.shape
act=env.action_space.n
state = env.reset()

"""Set specific value for random seed"""

#seed
RANDOM_SEED = 379
env.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)

"""Create agent according to desirable algorithm
In case of use prioritized replay buffer set beta value!=0 when agent is initialized
Moreover set hyperparameters memory size, gamma, lr, update frequency

"""

agent=Agent(state_shape,act,RANDOM_SEED,memory_size=500000,gamma=0.99,lr=1.5e-4,beta=0,update_frequency=4,norm=False,double=True,duel=False,noisy=False)
print(agent.optimizer,agent.steps)
#show agent's Qnetwork
agent.net

"""Training phase of the agent for specific training_steps with
 hyperparameters:eps_sart,eps_end,eps_decay,target_update,

*   eps_sart, eps_end, eps_decay: for eps_greedy policy
*   beta_start,beta_end,beta_decay: for importance of buffer's priorities
*   min_frames, max_frames, batch_size, target_update: conditions for network update
The agent will be trained for 50M steps, if there is no time limit
Additionally save net weights in selected location and track metrics for tensorboard
Finally 
"""

eps_start=1.0
eps_end=0.01
eps_decay=1000000
if agent.beta!=0:
  beta_start = 0.4
  beta_decay=10000000
  beta_end=1.0
target_update = 10000
save_net_weights=200000
steps_done= 0
start=time.time()
start1=start
start2=start
max_frames=105000
min_frames = 50000
batch_size=32
tot_rew= []
start=time.time()
steps_prev = 0
writer = SummaryWriter()
while steps_done<50000000:
    loss=0
    state = env.reset()
    score=0
    for j in range(max_frames):
        eps = max(eps_end, eps_start -steps_done / eps_decay)
        if agent.beta!=0:
          agent.beta = min(beta_end, beta_start + steps_done * (1.0 - beta_start) / beta_decay)
        steps_done += 1
        oldstate=state
        action=agent.select_action(oldstate,eps)
        state, reward, done, _ = env.step(action)
        score += reward
        loss=loss+agent.step(oldstate,action,reward,state,done,batch_size,min_frames) 
        if (steps_done % target_update == 0):
            agent.t_net.load_state_dict(agent.net.state_dict()) 
            if (steps_done % save_net_weights==0):
              s1="/content/drive/My Drive/thesis/classic_agent/prior_double_DQN_agent_v0"
              s2=str(int(steps_done/100000))
              s3=".pth"
              torch.save(agent.net.state_dict(), s1+s2+s3)
        if done:
            tot_rew.append(score)
            writer.add_scalar("loss", loss, steps_done)
            #plot every 200 epeisodes steps, epeisodes that agent done,average score of last 100 epeisodes,speed
            if(len(tot_rew)%200==0):
              speed=(steps_done-steps_prev)/(time.time()-start)
              start=time.time()
              steps_prev=steps_done
              mean_rew=np.mean(tot_rew[-100:])
              writer.add_scalar("epsilon", eps, steps_done)
              writer.add_scalar("speed", speed, steps_done)
              writer.add_scalar("reward_100", mean_rew, steps_done)
              writer.add_scalar("score", score, steps_done)
              print("%d:done %d episodes %.3f mean_rew %.2f fps" %(steps_done,len(tot_rew),mean_rew,,speed))
            break 
    env.close()
writer.close()


