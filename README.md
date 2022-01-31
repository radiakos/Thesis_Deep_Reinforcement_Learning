# Deep_Reinforcement_Learning_atari_gym

This repository consists my thesis's project in which I implement several deep q-learning algorithms in MsPacman game environment.

**Project's environment description**

The aim of this project is to train an agent in order to achieve the highest possible score. OpenAI gym provides the environment, in which 
agents are trained and evaluated.

**Environment's states**

The original agent's observation is an RGB image of the screen, which is an array of shape (210, 160, 3). However, a single image doesn't provide
agent with information about environment physics. For this reason, each observation consists of 4 stacked frames. Moreover, in order to reduce the 
complexity of the problem and remove unnecessary information from input, an preprocessing step is implemented. 
Finally, environment's states is an array of shape (4,84,84).

**Environment's actions**

The agent is able to take one of the 9 discrete joystick's actions.

**Environment's rewards**

The agent explores the environment according to his actions. There are several rewards that agent is being given accordingly to game rules.
Most common reward is given when the agent eats a pellet.

**Installation**


 1. Install the following libraries (in case u use anaconda and jupyter notebook is preinstalled)
```
	$ pip install numpy gym matplotlib torch tensorflow math random time cv2
```
**Run**
 

 1. Run jupyter notebook in cmd and navigate to the main directory
```
	$ jupyter notebook
```
 
