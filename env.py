# -*- coding: utf-8 -*-
# @Author: iori
# @Date:   2016-11-17 13:56:19
# @Last Modified by:   lei gao
# @Last Modified time: 2017-11-01 14:12:27

from __future__ import division
import heapq
import numpy as np
import copy
from collections import defaultdict
import random
import math

import logging
logger = logging.getLogger("APP.WORLD")

class Env(object):
	"""docstring for World"""
	def __init__(self, env_builder, temperature=1.0):
		super(Env, self).__init__()
		self.builder = env_builder
		# self.config = env_builder.env_config
		self.distances = env_builder.distances

	def reset(self):
		self.ordering = 0
		self.car = [0,0]
		# self.grid = np.zeros([self.config.screen_width,self.config.screen_height])
		# self.barrier = np.zeros([self.config.screen_width,self.config.screen_height])
		self.grid = np.zeros([10,10])
		self.barrier = np.zeros([10,10])
		for loc in [[2,2],[2,5],[2,8],[5,2],[5,5],[5,8],[8,2],[8,5],[8,8]]:
			self.barrier[loc[0],loc[1]] = 1

		for _ in range(self.config.job_num):
			coord = np.random.randint(self.config.screen_width,size=2)
			while self.grid[coord[0],coord[1]] != 0 or list(coord) == self.car or self.barrier[coord[0],coord[1]] == 1:
				coord = np.random.randint(self.config.screen_width,size=2)
			self.grid[coord[0],coord[1]] = np.random.randint(12, self.config.init_value+1)

		return self.get_state(), 0, False

	def get_state(self):
		car_state = np.zeros([self.config.screen_width,self.config.screen_height])
		car_state[self.car[0],self.car[1]] = 1
		job_state = self.grid / self.config.init_value
		barrier_state = self.barrier

		return np.stack([car_state, job_state, barrier_state],axis=-1)

	def take_action(self,policy):
		if policy == 0 and self.car[0] < self.config.screen_height-1:
			self.car[0] += 1
		elif policy == 1 and self.car[0] > 0:
			self.car[0] -= 1
		elif policy == 2 and self.car[1] < self.config.screen_width-1:
			self.car[1] += 1
		elif policy == 3 and self.car[1] > 0:
			self.car[1] -= 1

	def step(self,policy):
		self.ordering += 1

		# get actions
		self.take_action(policy)

		reward, count = 0, 0
		# get reward if possible
		if self.grid[self.car[0], self.car[1]]:
			reward += self.grid[self.car[0], self.car[1]]
			self.grid[self.car[0], self.car[1]] = 0.0
			count += 1
		if self.barrier[self.car[0],self.car[1]] == 1:
			reward -= 100

		# get new states of parcels
		for i in range(self.config.screen_width):
			for j in range(self.config.screen_height):
				if self.grid[i,j]:
					self.grid[i,j] -= 1
					if self.grid[i,j] == 0:
						count += 1

		# add new jobs
		for _ in range(count):
			coord = np.random.randint(self.config.screen_width,size=2)
			while self.grid[coord[0],coord[1]] != 0 or list(coord) == self.car or self.barrier[coord[0],coord[1]] == 1:
				coord = np.random.randint(self.config.screen_width,size=2)
			self.grid[coord[0],coord[1]] = np.random.randint(12, self.config.init_value+1)

		# check whether to terminate
		if self.ordering <= self.config.ticks:
			terminal = False
		else:
			terminal = True

		return  self.get_state(), reward/self.config.ticks, terminal
