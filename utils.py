import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#####################################################
#################### Transformer ####################
#####################################################

class CustomScaler(StandardScaler):
	def __init__(self, clamp = 3.0):
		super().__init__()
		self.clamp = clamp

	def transform(self, X):
		X = np.copy(X) - self.mean_
		X /= self.scale_
		X[X > self.clamp] = self.clamp
		X[X < -1 * self.clamp] = -1 * self.clamp
		return X

######################################################
################## General functions #################
######################################################

def print_dict(dict_inst, idx = 1):
	for key, value in dict_inst.items():
		if(isinstance(value, dict)):
			print('\t'*(idx-1), key, ': ')
			print_dict(value, idx = idx+1)
		else:
			print('\t'*idx, key, ': ', end = '')
			if(isinstance(value, np.ndarray)):
				print(value.shape)
			else: print(value)

def is_binary(mat):
	return len(np.unique(mat[:])) == 2