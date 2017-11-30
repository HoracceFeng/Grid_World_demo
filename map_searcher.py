import matplotlib.pyplot as plt
import matplotlib.animation
#from IPython.display import HTML

import os
import numpy as np



## The Grid World Map Searcher

## ==============================================
## World Parameters
### The Grid
PLANT_VALUE = -10
GOAL_VALUE = 1
EDGE_VALUE = -10
VISIBLE_RADIUS = 1  ## SET INPUT
GRID_SIZE = 10    
N_PLANTS = 15
N_GOALS = 12
STARTER = (1,1)

STEP_WASTE = -0.02

## The Agent
START_HEALTH = 1
STEP_VALUE = -0.03
## ==============================================

### The Grid

class Grid():
    def __init__(self, grid_size=GRID_SIZE, n_plants=N_PLANTS, n_goals=N_GOALS):
        self.grid_size = grid_size
        self.n_plants = n_plants
        self.n_goals = n_goals
        self.goal_get = 0

    def reset(self):
        padded_size = self.grid_size + 2 * VISIBLE_RADIUS
        self.grid = np.zeros((padded_size, padded_size)) # Padding for edges
        self.plant_set = []
        self.goal_set = []

        # Edges
        self.grid[0:VISIBLE_RADIUS, :] = EDGE_VALUE
        self.grid[-1*VISIBLE_RADIUS:, :] = EDGE_VALUE
        self.grid[:, 0:VISIBLE_RADIUS] = EDGE_VALUE
        self.grid[:, -1*VISIBLE_RADIUS:] = EDGE_VALUE

        # Randomly placed plants
        for i in range(self.n_plants):
            # plant_value = random.random() * (MAX_PLANT_VALUE - MIN_PLANT_VALUE) + MIN_PLANT_VALUE
            plant_value = PLANT_VALUE
            ry = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            rx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            self.grid[ry, rx] = plant_value
            self.plant_set.append((ry, rx))

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

        # Goal produce randomly
        for j in range(self.n_goals):
            goal_value = GOAL_VALUE
            gy = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            gx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            if (gy, gx) not in self.plant_set:
                self.grid[gy, gx] = goal_value
                self.goal_set.append((gy, gx))
            else:
                continue

        return self.grid


    def update(self, update):
        # Goal Value drop when time pass
        self.get = 0 
        if update == True:
            for _, goal in enumerate(self.goal_set):
                if self.grid[goal[0],goal[1]] == 0:
                    self.get += 1
                elif self.grid[goal[0],goal[1]] > 0:   
                    self.grid[goal[0],goal[1]] += STEP_WASTE

        ## auto add goal
        if self.get >= 3 and self.get <= 6:
            goal_value = GOAL_VALUE
            gy = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            gx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            self.grid[gy, gx] = goal_value
            self.goal_set.append((gy, gx))

        self.goal_get = self.get
        # print('Get Goal: ', self.get)
        return self.get
        

    def visible(self, pos):
        y, x = pos
        return self.grid[y-VISIBLE_RADIUS:y+VISIBLE_RADIUS+1, x-VISIBLE_RADIUS:x+VISIBLE_RADIUS+1]




class Searcher():
	def __init__(self, grid_size=GRID_SIZE, )







