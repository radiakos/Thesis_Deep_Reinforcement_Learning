# Deep_reinforcement_learning_navigation

This repository consists a project in which I implement the deep q-learning algorithm in Unity's Banana environment.

**Project's environment description**

The aim of this project is to train an agent in order to collect bananas. Unity ml-agents provide the environment for this project. The environment is concsiderd solved when the agent receives an average score of +13 over 100 consecutive episodes.

**Environment's states**

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction

**Environment's actions**

The agent is able to take one of the following four discrete actions: 
 1. move backward
 2. move backward 
 3. turn left
 4. turn right



**Environment's rewards**

The agent explores the environment according to his actions. There is a reward in case he finds a banana. More specifically, when he finds a banana receives a reward of +1 if it is yellow or -1 if it is blue.

**Installation**

 1.  Download, unzip and setup the Unity environment in this project folder
 
     -  Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
     -   Mac OSX:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
     -   Windows (32-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
     -   Windows (64-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
 2. Install the following libraries (in case u use anaconda and jupyter notebook is preinstalled)
```
	$ pip install numpy unityagents matplotlib torch
```
**Run**
 
 
 1. Run jupyter notebook in cmd and navigate to the main directory
```
	$ jupyter notebook
```
 
