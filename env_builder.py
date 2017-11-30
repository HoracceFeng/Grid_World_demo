# -*- coding: utf-8 -*-
# @Author: ioriiod0
# @Date:   2017-06-13 13:44:29
# @Last Modified by:   lei gao
# @Last Modified time: 2017-08-26 22:05:43

import random
import logging
import numpy as np
from collections import defaultdict
from .env import *
logger = logging.getLogger('APP.ENV')

class EnvBuilder(object):
	"""docstring for EnvBuilder"""

	def __init__(self, args):
		super(EnvBuilder, self).__init__()
		self.distances = dict()
		self.env_config = args
		self.setup()

	def setup(self):
		for i in range(self.env_config.screen_width):
			for j in range(self.env_config.screen_height):
				for k in range(self.env_config.screen_width):
					for l in range(self.env_config.screen_height):
						od = (i, j, k, l)
						self.distances[od] = abs(i-k) + abs(j-l)


	def build_env(self, temperature):
		return Env(self, temperature)
