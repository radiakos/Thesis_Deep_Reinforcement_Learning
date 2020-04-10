# deep_reinforcement_learning_navigation

This repository consists a project in which i implement deep q-learning algorithm in unity's Banana environment.

Project's environment description

The aim of this project is to train an agent in order to collect bananas. Unity ml-agents provide the environment
for this project. The environment is concsiderd solved when the agent receives an average score of +13 over 100 
consecutive episodes.

Environment's states

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction

Environment's actions

The agent is able to take one of the following four discrete actions:
0)move forward
1)move backward
2)turn left
3)turn right

Environment's rewards

The agent explores the environment according to his actions. There is a reward in case he finds a banana.
More specifically, when he finds a banana receives a reward of +1 if it is yellow or -1 if it is blue.




