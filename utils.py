import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class ZSLPipeline(Pipeline):
	def __init__(self, steps):
		super().__init__(steps)

	def _transform(self, X):
		for _, step in self.steps[:-1]:
			X = step.transform(X)
		return X

	def fit(self, X, S, y):
		for _, step in self.steps[:-1]:
			X = step.fit_transform(X)
		self.steps[-1][1].fit(X, S, y)
		return self

	def decision_function(self, X, S):
		return self.steps[-1][1].decision_function(self._transform(X), S)

	def predict(self, X, S):
		return self.steps[-1][1].predict(self._transform(X), S)

	def score(self, X, S, y):
		return self.steps[-1][1].score(self._transform(X), S, y)

#####################################################
#################### Transformer ####################
#####################################################

class ClampScaler(TransformerMixin):
	def __init__(self, clamp = 3.0):
		super().__init__()
		self.clamp = clamp

	def fit(self, X, y = None):
		return self
	
	def transform(self, T):
		T[T > self.clamp] = self.clamp
		T[T < -1 * self.clamp] = -1 * self.clamp
		return T

class CustomScaler(StandardScaler):
	def __init__(self, clamp = 3.0):
		super().__init__()
		self.clamp = clamp

	def transform(self, X):
		T = super().transform(X)
		T[T > self.clamp] = self.clamp
		T[T < -1 * self.clamp] = -1 * self.clamp
		return T

class UnitScaler(BaseEstimator, TransformerMixin):
	def __init__(self):
		super().__init__()

	def _normalize(self, M, axis = 0):
		return M / (np.linalg.norm(M, axis = axis, keepdims=True) + 1e-10)

	def fit(self, X, y = None):
		self.norm_ = np.linalg.norm(X, axis = 0, keepdims=True)
		return self

	def transform(self, X):
		return X / self.norm_

	def inverse_transform(self, X): 
		return X * self.norm_

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)

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

def print_dstruct(data):
	print('######################')
	print('### Seen Classes ###')
	print('Seen data input: ', data['seen_data_input'].shape)
	print('Seen data output: ', data['seen_data_output'].shape)
	print('Seen attribute matrix:', data['seen_attr_mat'].shape)
	print('Seen class IDs:', data['seen_class_ids'].shape)

	print('### Unseen Classes ###')
	print('Uneen data input: ', data['unseen_data_input'].shape)
	print('Unseen data output: ', data['unseen_data_output'].shape)
	print('Unseen attribute matrix:', data['unseen_attr_mat'].shape)
	print('Unseen class IDs:', data['unseen_class_ids'].shape)
	print('######################\n')

def is_binary(mat):
	return len(np.unique(mat[:])) == 2

def normalize(M, axis = 0):
	return M / (np.linalg.norm(M, axis = axis, keepdims=True) + 1e-10)