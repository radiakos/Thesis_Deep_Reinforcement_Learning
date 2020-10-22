import gym
import random
from random import randrange
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from atari_wrappers import wrap_deepmind
from agent_evaluate import Agent
from replay_buffer import replay_buffer
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

"""
Create environment and agent according to the trained agent
Select random seed value
Select env's name 
Select agent's Qnet
Set specific value for random seed
"""
#create env according to deepmind preprocessing
RANDOM_SEED=137
env = gym.make('MsPacmanNoFrameskip-v0')
env = wrap_deepmind(env,episode_life=False,clip_rewards=False)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
env.action_space.seed(RANDOM_SEED)
state_shape=env.observation_space.shape
act=env.action_space.n
state = env.reset()
#create agent 
agent=Agent(state_shape,act,norm=False,duel=True,noisy=False)
#use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""
Params to modify
======
Num_of_eps(int): number of episodes per evaluation
Max_frames(int): number of max frames per epeisode
eps(float): eps for eps-greedy policy. According to paper revisiting the ale
this value must be equal to 0.01 during evaluation
x_axis(float):determine the x-axis of tensorboard graphs. For 20 hours of training
the value must be equal to 20/(20+1)
"""
Num_of_eps=100
Max_frames=105000
eps=0.01
x_axis=20/21
"""
Specify agent's Qnetwork and his weight's path
params to modify
======
path0,path1:specify weight's path
agent_t_steps_max(int):max number of available agent's weights
"""
#create agent 
agent=Agent(state_shape,act,norm=False,duel=False,noisy=False)
path0="/content/drive/My Drive/thesis/classic_agent/"
path1="prior_double_DQN_agent_v0"
agent_t_steps_max=70

"""
Evaluate once the agent and save metrics in tensorboard's events with the
use of a SummaryWriter
params to modify
======
"""

all_rewards=[]
speed=0
time0=0
for p in range(1):
    path3=".pth"
    sr=path1+"100eps_test"
    writer = SummaryWriter(comment=sr)
    for l in range(20):
        show_step=70/20
        k=int(show_step*l+2)
        if(k%2==1): k+=1
        path2=str(k)
        s=path0+path1+path2+path3
        agent.net.load_state_dict(torch.load(s, device))
        agent.net.eval()
        #eps-greedy policy
        steps_done=0
        tot_rew= []
        start=time.time()
        steps_prev=1
        for i in range(Num_of_eps):
            state = env.reset()
            score=0
            for j in range(Max_frames):
                steps_done += 1
                oldstate=state
                action=agent.select_action(state,eps)
                state, reward, done, _ = env.step(action)
                if done:
                  tot_rew.append(score)
                  #print(score)
                  break
                #for watching agent live uncomment next line
                #env.render()
                score += reward
            env.close()
        time0=time.time()-start
        speed=(steps_done-steps_prev)/(time0)
        start=time.time()
        steps_prev=steps_done
        av_eps_steps=steps_done/Num_of_eps
        r=np.array(tot_rew)
        r.sort()
        t=int(l/x_axis+1)
        max=np.mean(np.max(r[-10:]))
        min=np.mean(np.min(r[:10]))
        mean=np.nanmean(tot_rew)
        all_rewards.append(tot_rew)
        writer.add_scalar("J/hours", mean, t)
        writer.add_scalar("max_score/hours", max, t)
        writer.add_scalar("min_score/hours", min, t)
        writer.add_scalar("speed/hours", speed, t)
        writer.add_scalar("av_eps_steps/hours", av_eps_steps, t)
        writer.add_scalar("speed/hours", speed, t)
        writer.add_scalar("J/training_steps(*100K)", mean, k)
        writer.add_scalar("max_score/training_steps(*100K)", max, k)
        writer.add_scalar("min_score/training_steps(*100K)", min, k)
        writer.add_scalar("speed/training_steps(*100K)", speed, k)
        writer.add_scalar("av_eps_steps/training_steps(*100K)", av_eps_steps, k)
        writer.add_scalar("speed/training_steps(*100K)", speed, k)
    writer.close()






