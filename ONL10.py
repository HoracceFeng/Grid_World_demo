#!/usr/bin/env python

# # Practical PyTorch: Playing GridWorld with Reinforcement Learning (Actor-Critic with REINFORCE)

# ## Resources

# ## Requirements

#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
#from IPython.display import HTML

import os
import numpy as np
import sconce
#from itertools import count
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from helpers import *

from __future__ import division
import heapq
import copy
from collections import defaultdict

from env import *
from env_builder import *

# Configuration

gamma = 0.9 # Discounted reward factor

hidden_size = 50
learning_rate = 1e-4
weight_decay = 1e-5

log_every = 1000
render_every = 20000

job = sconce.Job('rl2', {
    'gamma': gamma,
    'learning_rate': learning_rate,
})
job.log_every = log_every
job.plot_every = 500

DROP_MAX = 0.3
DROP_MIN = 0.05
DROP_OVER = 200000

## The Grid World, Agent and Environment

## ==============================================
## World Parameters
### The Grid
PLANT_VALUE = -10
GOAL_VALUE = 5
EDGE_VALUE = -5
# VISIBLE_RADIUS = 10  ## SET INPUT 
GRID_SIZE = 10    
N_PLANTS = 15
N_GOALS = 12
UPDATE_THRESHOLD = 8
# STARTER = (VISIBLE_RADIUS, VISIBLE_RADIUS)
STARTER = (1,1)
STEP_WASTE = -0.2
WON_THRESHOLD = 100

#     ## MAPS
# file = open('maze.txt')
# maze = file.readlines()
# goal_list = []
# edge_list = []
# rout_list = []
# for i in range(len(maze)):
#     for j, val in enumerate(maze[i]):
#         if val == "%":
#             edge_list.append((i,j))
#         if val == ".":
#             goal_list.append((i,j))
#         if val == " ":
#             rout_list.append((i,j))
            
# print('Goal list len before: ', len(goal_list))

## The Agent
START_HEALTH = 10
STEP_VALUE = -0.1
## ==============================================

### The Grid





# ## Actor-Critic network

class Policy(nn.Module):
    def __init__(self, hidden_size):
        super(Policy, self).__init__()

        # visible_squares = (VISIBLE_RADIUS * 2 + 1) ** 2
        # input_size = visible_squares + 1 + 2 # Plus agent health, y, x
        # input_size = 10*10 + 2 # Plus y, x
        input_size = 12*12 + 2

        self.inp = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 4 + 1, bias=False) # For both action and expected value

    def forward(self, x):
        # print('X.before', x.size())
        x = x.view(1, -1)
        # print('X.shape ', x.size())
        x = F.tanh(x) # Squash inputs
        x = F.relu(self.inp(x))
        x = self.out(x)

        # Split last five outputs into scores and value
        scores = x[:,:4]
        value = x[:,4]
        return scores, value

# ## Selecting actions

def select_action(e, state):
    drop = interpolate(e, DROP_MAX, DROP_MIN, DROP_OVER)

    # print('State.shape_before ', len(state))
    state = Variable(torch.from_numpy(state).float())
    # print('State.shape_after ', state.size())
    scores, value = policy(state) # Forward state through network
    scores = F.dropout(scores, drop, True) # Dropout for exploration
    scores = F.softmax(scores)
    action = scores.multinomial() # Sample an action

    return action, value

# ## Playing through an episode

def run_episode(e):
    state = env.reset()
    # print('RE_state: ', len(state))
    actions = []
    values = []
    rewards = []
    step_count = 0
    done = False
    # print('run in')

    while not done:
        # print('RE_e:%d  RE_state_in' % e, len(state))
        step_count += 1
        action, value = select_action(e, state)
        state, reward, done = env.step(action.data[0, 0], step_count=step_count, update=True)    ## online
        # state, reward, done = env.step(action.data[0, 0], update=True)    ## offline
        # print("step ahead")
        actions.append(action)
        values.append(value)
        rewards.append(reward)

    print('step count: ', step_count)
    return actions, values, rewards

# ## Using REINFORCE with a value baseline

mse = nn.MSELoss()

def finish_episode(e, actions, values, rewards):

    # Calculate discounted rewards, going backwards from end
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.Tensor(discounted_rewards)

    # Use REINFORCE on chosen actions and associated discounted rewards
    value_loss = 0
    for action, value, reward in zip(actions, values, discounted_rewards):
        reward_diff = reward - value.data[0] # Treat critic value as baseline
        action.reinforce(reward_diff) # Try to perform better than baseline
        value_loss += mse(value, Variable(torch.Tensor([reward]))) # Compare with actual reward

    # Backpropagate
    optimizer.zero_grad()
    nodes = [value_loss] + actions
    gradients = [torch.ones(1)] + [None for _ in actions] # No gradients for reinforced values
    autograd.backward(nodes, gradients)
    optimizer.step()

    # Save Model
    if e % 10000 == 0:
        ckpt = 'out_checkpoint/RG10_' + str(e) + '.pkl'
        torch.save(policy.state_dict(), ckpt)

    return discounted_rewards, value_loss


def animate(history, filename):
    frames = len(history)
    print("Rendering %d frames..." % frames)
    fig = plt.figure(figsize=(6, 2))
    fig_grid = fig.add_subplot(121)
    fig_health = fig.add_subplot(243)
    # fig_visible = fig.add_subplot(244)
    fig_get = fig.add_subplot(247)

    fig_health.set_autoscale_on(False)
    fig_get.set_autoscale_on(False)
    health_plot = np.zeros((frames, 1))
    get_plot = np.zeros((frames, 1))

    def render_frame(i):
        # grid, visible, health = history[i]
        grid, health, get, pos = history[i]
        # Render grid
        fig_grid.matshow(grid, vmin=-1, vmax=1, cmap='jet')
        # fig_visible.matshow(visible, vmin=-1, vmax=1, cmap='jet')
        # Render health chart
        health_plot[i] = health
        fig_health.clear()
        fig_health.axis([0, frames, 0, 50])
        fig_health.plot(health_plot[:i + 1])
        # Render goal-get chart
        get_plot[i] = get
        fig_get.clear()
        fig_get.axis([0, frames, 0, 50])
        fig_get.plot(get_plot[:i + 1])

    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100
    )

    plt.close()
    anim.save(filename)
#    display(HTML(anim.to_html5_video()))




if __name__ == '__main__':
    
    # VISIBLE_RADIUS = 10
    STARTER = (1, 1)

    hidden_size = 50
    learning_rate = 1e-3
    weight_decay = 1e-4
    
    log_every = 100
    render_every = 4000

    value_avg_list = []
    round_list = []

    #     ## MAPS
    # file = open('maze.txt')
    # maze = file.readlines()
    # goal_list = []
    # edge_list = []
    # rout_list = []
    # for i in range(len(maze)):
    #     for j, val in enumerate(maze[i]):
    #         if val == "%":
    #             edge_list.append((i,j))
    #         if val == ".":
    #             goal_list.append((i,j))
    #         if val == " ":
    #             rout_list.append((i,j))
                
    # print('Goal list len before: ', len(goal_list))

    builder = EnvBuilder(conf)
    env = Env()
    policy = Policy(hidden_size=hidden_size)

    # ckpt_path = 'out_checkpoint/RG10_190000.pkl'
    # policy.load_state_dict(torch.load(ckpt_path))
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)#, weight_decay=weight_decay)
    
    reward_avg = SlidingAverage('reward avg', steps=log_every)
    value_avg = SlidingAverage('value avg', steps=log_every)
    
    out_doc = 'out_GR'
    



    # if os.path.exists(out_doc):
    #     os.mkdir(out_doc)
    
    e = 0
    print('>>>>>>>>> Start to play ...')

    while reward_avg < 0.75:
        actions, values, rewards = run_episode(e)
        final_reward = rewards[-1]

        # print "==========================================================="
        # print('Games        : ', e)
        # print('Actions shape: ', len(actions))
        # print('Values shape:  ', len(values))
        # print('Rewards shape: ', len(rewards))
        
        discounted_rewards, value_loss = finish_episode(e, actions, values, rewards)
        
        reward_avg.add(final_reward)
        value_avg.add(value_loss.data[0])
        
        filename = os.path.join(out_doc , str(e) + ".mp4")  

        value_avg_list.append(value_avg.value)
        round_list.append(e)  
        
        if e % log_every == 0:
            print('[epoch=%d]  reward_avg:%.3f  value_avg:%.3f  final_reward:%.3f' \
                % (e, reward_avg.value, value_avg.value, final_reward))
            # print reward_avg.total()
        
        if e > 0 and e % render_every == 0:
            animate(env.history, filename)
            plt.plot(round_list, value_avg_list)
            plot_name = os.path.join(out_doc, str(e)+".jpg")
            plt.savefig(plot_name)

        e += 1



    # while reward_avg < 0.75:
    #     try:
    #         # print('round start')
    #         actions, values, rewards = run_episode(e)
    #         final_reward = rewards[-1]

    #         # print "==========================================================="
    #         # print('EPOCH        : ', e)
    #         # print('Actions shape: ', len(actions))
    #         # print('Values shape:  ', len(values))
    #         # print('Rewards shape: ', len(rewards))
            
    #         discounted_rewards, value_loss = finish_episode(e, actions, values, rewards)
            
    #         # print('round end')

    #         reward_avg.add(final_reward)
    #         value_avg.add(value_loss.data[0])
            
    #         filename = os.path.join(out_doc , str(e) + ".mp4")    
            
    #         if e % log_every == 0:
    #             print('[Rounds=%d]  reward_avg:%.3f  value_avg:%.3f  final_reward:%.3f' \
    #                 % (e, reward_avg.value, value_avg.value, final_reward))
    #             # print reward_avg.total()
            
    #         if e > 0 and e % render_every == 0:
    #             animate(env.history, filename)
            
    #         e += 1
    #         # print('Next Round ..')

    #     except:
    #         e += 1
    #         # print('round skip')
    #         continue
















