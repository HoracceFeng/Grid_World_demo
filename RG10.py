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
VISIBLE_RADIUS = 10  ## SET INPUT 
GRID_SIZE = 10    
N_PLANTS = 15
N_GOALS = 12
UPDATE_THRESHOLD = 8
STARTER = (VISIBLE_RADIUS, VISIBLE_RADIUS)
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

class Grid():
    # def __init__(self, edge_list=edge_list, goal_list=goal_list, rout_list=rout_list, grid_size=GRID_SIZE, n_plants=N_PLANTS, n_goals=N_GOALS):
    def __init__(self, grid_size=GRID_SIZE, n_plants=N_PLANTS, n_goals=N_GOALS):
        # self.edge_list = edge_list
        # self.goal_list = goal_list
        # self.rout_list = rout_list
        self.grid_size = grid_size
        self.n_plants = n_plants
        self.n_goals = n_goals
        self.get_list = []
        self.goal_get = 0

    def reset(self):
        padded_size = self.grid_size + 2 * VISIBLE_RADIUS
        self.grid = np.zeros((padded_size, padded_size))           ## padding version -- Padding for edges
        # self.grid = np.zeros((self.grid_size, self.grid_size))   ## no padding version

        # Map
        self.file = open('maze.txt')
        self.maze = self.file.readlines()
        self.goal_list = []
        self.edge_list = []
        self.rout_list = []
        for i in range(len(self.maze)):
            for j, val in enumerate(self.maze[i]):
                if val == "%":
                    self.edge_list.append((i,j))
                if val == ".":
                    self.goal_list.append((i,j))
                if val == " ":
                    self.rout_list.append((i,j))


        # Edges
        self.grid[0:VISIBLE_RADIUS, :] = EDGE_VALUE
        self.grid[-1*VISIBLE_RADIUS:, :] = EDGE_VALUE
        self.grid[:, 0:VISIBLE_RADIUS] = EDGE_VALUE
        self.grid[:, -1*VISIBLE_RADIUS:] = EDGE_VALUE

        # Map
        for _, edge_point in enumerate(self.edge_list):
            self.grid[edge_point[0]+VISIBLE_RADIUS, edge_point[1]+VISIBLE_RADIUS] = EDGE_VALUE


        for _, goal_point in enumerate(self.goal_list):
            self.grid[goal_point[0]+VISIBLE_RADIUS, goal_point[1]+VISIBLE_RADIUS] = GOAL_VALUE

        # print('after goal list:', len(self.goal_list))

        # # Randomly placed plants
        # for i in range(self.n_plants):
        #     # plant_value = random.random() * (MAX_PLANT_VALUE - MIN_PLANT_VALUE) + MIN_PLANT_VALUE
        #     plant_value = PLANT_VALUE
        #     ry = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
        #     rx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
        #     self.grid[ry, rx] = plant_value
        #     self.plant_set.append((ry, rx))

        # # Goal in one of the corners
        # S = VISIBLE_RADIUS
        # E = self.grid_size + VISIBLE_RADIUS - 1
        # gps = [(E, E), (S, E), (E, S), (S, S)]
        # gp = gps[random.randint(0, len(gps)-1)]
        # self.grid[gp] = GOAL_VALUE

        # # Goal produce randomly
        # for j in range(self.n_goals):
        #     goal_value = GOAL_VALUE
        #     gy = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
        #     gx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
        #     if (gy, gx) not in self.plant_set:
        #         self.grid[gy, gx] = goal_value
        #     else:
        #         continue

        # # Goal produce randomly
        # for j in range(self.n_goals):
        #     goal_value = GOAL_VALUE
        #     gy = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
        #     gx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
        #     if (gy, gx) not in self.plant_set:
        #         self.grid[gy, gx] = goal_value
        #         self.goal_set.append((gy, gx))
        #     else:
        #         continue

        return self.grid
    
    ## fix grid world update
    def update(self, update):
        # self.get = 0
        goal_bk = self.goal_list
        if update == True:
            for index, goal in enumerate(self.goal_list):
                if self.grid[goal[0]+VISIBLE_RADIUS, goal[1]+VISIBLE_RADIUS] == 0:
                    goal_bk.pop(index)
                    self.rout_list.append(goal)
                    self.get_list.append(goal)
                elif self.grid[goal[0]+VISIBLE_RADIUS, goal[1]+VISIBLE_RADIUS] > 0:
                    self.grid[goal[0]+VISIBLE_RADIUS,goal[1]+VISIBLE_RADIUS] += STEP_WASTE
                else:
                    continue

            self.goal_list = goal_bk
            self.goal_get = len(self.get_list)

            if len(self.goal_list) <= 10:
                for time in range(random.randint(0,3)):
                    new_goal_index = random.randint(0, len(self.rout_list))
                    new_goal = self.rout_list.pop(new_goal_index)
                    self.grid[new_goal[0]+VISIBLE_RADIUS, new_goal[1]+VISIBLE_RADIUS] = GOAL_VALUE
                    self.goal_list.append(new_goal)

    ## random grid world update
     # def update(self, update):
     #     # Goal Value drop when time pass
     #     self.get = 0 
     #     if update == True:
     #         for _, goal in enumerate(self.goal_set):
     #             if self.grid[goal[0],goal[1]] == 0:
     #                 self.get += 1
     #             elif self.grid[goal[0],goal[1]] > 0:   
     #                 self.grid[goal[0],goal[1]] += STEP_WASTE

     #     rest_score = self.grid.sum() + ((GRID_SIZE+VISIBLE_RADIUS*2)**2 - GRID_SIZE**2 + PLANT_VALUE)*abs(EDGE_VALUE) - GOAL_VALUE * N_GOALS
     #     ## auto add goal
     #     if rest_score < UPDATE_THRESHOLD:
     #     # if self.get > 3 and self.get < 6:
     #         goal_value = GOAL_VALUE
     #         gy = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
     #         gx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
     #         if (gy, gx) not in self.plant_set and (gy, gx) not in self.goal_set:
     #             self.grid[gy, gx] = goal_value
     #             self.goal_set.append((gy, gx))

    #     self.goal_get = self.get
    #     # print('Get Goal: ', self.get)
    #     return self.get
        

    # def visible(self, pos):
    #     y, x = pos
    #     return self.grid[y-VISIBLE_RADIUS:y+VISIBLE_RADIUS+1, x-VISIBLE_RADIUS:x+VISIBLE_RADIUS+1]

    def env_state(self):
        # print(self.grid[:,:])
        return self.grid[VISIBLE_RADIUS:VISIBLE_RADIUS+10, VISIBLE_RADIUS:VISIBLE_RADIUS+10]

# ### The Agent

# START_HEALTH = 1
# STEP_VALUE = -0.02

class Agent:
    def reset(self):
        self.health = START_HEALTH

    def act(self, action):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        y, x = self.pos
        if action == 0: y -= 1
        elif action == 1: x += 1
        elif action == 2: y += 1
        elif action == 3: x -= 1
        self.pos = (y, x)
        self.health += STEP_VALUE # Gradually getting hungrier

# ### The Environment

class Environment:
    def __init__(self, starter=None):
        self.grid = Grid()
        self.agent = Agent()
        self.starter = starter

    def reset(self):
        """Start a new episode by resetting grid and agent"""
        self.grid.reset()
        self.agent.reset()
        if self.starter == None:
            c = int(self.grid.grid_size / 2)
            self.agent.pos = (c, c)
        else:
            self.agent.pos = self.starter  

        self.t = 0
        self.rewards = []
        self.history = []
        self.record_step()

        return self.visible_state

    def record_step(self):
        """Add the current state to history for display later"""
        grid = np.array(self.grid.grid)
        grid[self.agent.pos] = self.agent.health * 0.5 # Agent marker faded by health
        # visible = np.array(self.grid.visible(self.agent.pos))
        # self.history.append((grid, visible, self.agent.health))
        # self.history.append((grid, visible, self.agent.health, self.grid.goal_get))
        self.history.append((grid, self.agent.health, self.grid.goal_get))

    @property
    def visible_state(self):
        """Return the visible area surrounding the agent, and current agent health"""
        # visible = self.grid.visible(self.agent.pos)
        visible = self.grid.env_state()
        # print('sss', visible.size())
        y, x = self.agent.pos
        yp = (y - VISIBLE_RADIUS) / self.grid.grid_size
        xp = (x - VISIBLE_RADIUS) / self.grid.grid_size
        # extras = [self.agent.health, yp, xp]
        extras = [yp, xp]
        return np.concatenate((visible.flatten(), extras), 0)

    # ## online ##
    # def step(self, action, step_count, reward, update=False):
    #     """Update state (grid and agent) based on an action"""
    #     self.agent.act(action)
    #     self.grid.update(update=update)

    #     # Get reward from where agent landed, add to agent health
    #     value = self.grid.grid[self.agent.pos]
    #     self.grid.grid[self.agent.pos] = 0
    #     self.agent.health += value

    #     # Online Rewards based on grid_value:
    #     if value != 0:
    #         reward += value
    #     if value == 0:
    #         reward += STEP_VALUE * 2

    #     if self.agent.health <= 0 or step_count >= 500:
    #         done = True

    #     # Save in history
    #     self.record_step()

    #     return self.visible_state, reward, done

    ## offline ##
    def step(self, action, step_count, update=False):
        """Update state (grid and agent) based on an action"""
        origin_agent_pos = self.agent.pos
        self.agent.act(action)
        value = self.grid.grid[self.agent.pos]
        # if value<= 0:
        #     self.agent.pos = origin_agent_pos
        self.grid.update(update=update)

        # Get reward from where agent landed, add to agent health
        if value != EDGE_VALUE:
            self.grid.grid[self.agent.pos] = 0
        else:
            self.grid.grid[self.agent.pos] = EDGE_VALUE
            self.agent.pos = origin_agent_pos
        self.agent.health += value

        # Offline Rewards - 1
        # Check if agent won (reached the goal) or lost (health reached 0)
        # won = value >= GOAL_VALUE * N_GOALS
        # won = value == GOAL_VALUE
        # won = value >= 100
        # lost = self.agent.health <= 0
        # if step_count >= 50 or lost:
        #     done = True
        # else:
        #     done = False

        # # Rewards at end of episode
        # if won:
        #     reward = 10
        # elif lost:
        #     reward = -10
        # else:
        #     reward = 0 # Reward will only come at the end
        #     # reward = value # Try this for quicker learning     ## slow down?

        # Offline Rewards - 2
        lost = self.agent.health <= 0
        if step_count >= 50 or lost:
            done = True
        else:
            done = False

        if value > 0:
            self.rewards.append(value)
            reward = value
        elif value < 0:
            self.rewards.append(value)
            reward = value * 2
        else:
            # self.rewards.append(STEP_WASTE * 2)
            # reward = STEP_WASTE * 2
            reward = -1

        # if self.agent.health <= 0 or step_count >= 500:
        #     done = True
        # else:
        #     done = False

        # # Offline Rewards - 3
        # if value != 0:
        #     self.rewards.append(value)
        # else:
        #     self.rewards.append(STEP_WASTE * 2)

        # if sum(self.rewards) > WON_THRESHOLD:
        #     won = True
        # else:
        #     won = False

        # if self.agent.health <= 0 or step_count >= 500:
        #     done = True
        #     if won == True:
        #         reward = sum(self.rewards)
        #     else:
        #         reward = - WON_THRESHOLD
        # else:
        #     done = False
        #     reward = value

        # Save in history
        self.record_step()

        return self.visible_state, reward, done



# ## Actor-Critic network

class Policy(nn.Module):
    def __init__(self, hidden_size):
        super(Policy, self).__init__()

        # visible_squares = (VISIBLE_RADIUS * 2 + 1) ** 2
        # input_size = visible_squares + 1 + 2 # Plus agent health, y, x
        input_size = 10*10 + 2 # Plus y, x

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
        grid, health, get = history[i]
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
    
    VISIBLE_RADIUS = 10
    STARTER = (VISIBLE_RADIUS, VISIBLE_RADIUS)

    hidden_size = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    log_every = 100
    render_every = 4000


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

    
    env = Environment(starter=STARTER)
    policy = Policy(hidden_size=hidden_size)

    # ckpt_path = 'out_checkpoint/RG10_190000.pkl'
    # policy.load_state_dict(torch.load(ckpt_path))
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)#, weight_decay=weight_decay)
    
    reward_avg = SlidingAverage('reward avg', steps=log_every)
    value_avg = SlidingAverage('value avg', steps=log_every)
    
    out_doc = 'out'
    



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
        
        if e % log_every == 0:
            print('[epoch=%d]  reward_avg:%.3f  value_avg:%.3f  final_reward:%.3f' \
                % (e, reward_avg.value, value_avg.value, final_reward))
            # print reward_avg.total()
        
        if e > 0 and e % render_every == 0:
            animate(env.history, filename)
        
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
















