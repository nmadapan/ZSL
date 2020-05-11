'''
	This file implements a zero-shot learning (ZSL) classifier 
	known as SAE (Semantic Auto Encoder) proposed by Kodriov et al.

	The class, SAE is implemented in a format similar to 
	standard scikit learn packages, except that, in addition to
	class labels, you should pass a semantic description (SD)
	matrix as well. 

	NOTE: You CAN NOT use this model directly with other scikit-
	learn packages such as sklearn.model_selection.GridSearchCV
	and sklearn.pipeline.Pipeline as scikit-learn has no good 
	way to pass both class labels and semantic description 
	matrix to methods such as predict(), score() etc. However, 
	you can pass additional arguments (using **kwargs) to
	fit() method. 

	Notations:
		* n - No. of instances of seen classes
		* d - Dimension of the data
		* a - No. of attributes/ descriptors
		* z - No. of seen classes
		* X (n x d) - Input data matrix
		* S (z x a) - SD matrix of seen classes
		* SS (n x a) - SD matrix of all seen instances
		* Y (n x z) - One hot output matrix
		* W (a x d) - Weight matrix

	Closed form solution using SAE:
		* $$ P W + W Q = R $$
		* $$ P = SS^T SS, Q = \lambda X^T X, R = (1+\lambda) SS^T X $$
		* P (a x a), Q (d x d), R (a x d), W (a x d)
		* Solve this Sylvester equation to get the solution. 

	Input to Attributes:
		* $$ S' = X W^T $$

	Input to classes:
		* $$ Y' = S X W^T $$

	Link to the paper:
		* https://arxiv.org/abs/1704.08345

	Author: Naveen Madapana
	Updated: 10 May 2020
'''

import numpy as np
import scipy
import scipy.io
from scipy.linalg import solve_sylvester
import argparse
import sys
from sklearn.base import BaseEstimator
from os.path import dirname, join
from sklearn.metrics import accuracy_score

class SAE(BaseEstimator):
	def __init__(self, lambdap = 5e5, rs = None, debug = False):
		'''
		Description:
			* This class inherits BaseEstimator which defines 
				get_params() and set_params() functions. 
		Input parameters:
			lambdap: Hyper parameter.
			rs: random seed
			debug: if True, print statements are activated
		Order in which GridSearchCV calls functions in scikit-learn:
			set_params() ==> fit() ==> score()
		'''

		self.lambdap = lambdap
		self.rs = rs
		self.debug = debug

		## Attributes: Created by fit()
		# self.S_ = None
		# self.W_ = None

	def _normalize(self, M, axis = 0):
		return M / (np.linalg.norm(M, axis = axis, keepdims=True) + 1e-10)

	def fit(self, X, S, y):
		'''
		Input arguments:
			* X (n x d or n x n): 2D np.ndarray - input matrix
			* S (z x a): 2D np.ndarray - semantic description matrix
			* y (n x 1): 1D np.ndarray of class label indices.
		Math:
			* $$ P W + W Q = R $$
			* $$ P = SS^T SS, Q = \lambda X^T X, R = (1+\lambda) SS^T X $$
			* P (a x a), Q (d x d), R (a x d), W (a x d)
			* Solve this Sylvester equation to get the solution. 		
		Attributes created:
			* S_ (SD matrix of seen classes)
			* W_ (weight matrix to transform inputs to SDs)
		Return:
			* self
		'''

		## Assertions here
		X = self._normalize(X)

		# S: z x a ==> n x a # X: n x d
		A = np.dot(S[y, :].T, S[y, :]) # a x a
		B = self.lambdap * np.dot(X.T, X) # d x d
		C = (1+self.lambdap) * np.dot(S[y, :].T, X) # a x d
		W = solve_sylvester(A,B,C) # a x d
		self.W_ = self._normalize(W, axis=1)
		self.S_ = S

	def decision_function(self, X, S):
		'''
		Input arguments:
			* X (n' x d or n' x n'): 2D np.ndarray - input matrix for test data
			* S (z' x a): 2D np.ndarray - semantic description matrix of test classes
		Math:
			Input to Attributes:
				* $$ S' = X W^T $$
			Input to classes:
				* $$ Y' = S X W^T $$		
		Return:
			* Z: (n' x z') matrix of class scores
		'''		

		## Assert to call fit first()
		try:
			_ = self.W_
		except AttributeError as exp:
			print('Error! W_ attribute does not exist. Run fit() first. ')
			raise exp
		
		S_pred = np.dot(X, self.W_.T) # N x a
		# Normalize for cosine similarity
		S_pred = self._normalize(S_pred, axis = 1)

		# Normalize each attribute first
		S = self._normalize(S, axis = 0)

		# Normalize each class now # For cosine similarity
		S = self._normalize(S, axis = 1)

		# Class probabilities
		Z = np.dot(S_pred, S.T)

		return Z

	def predict(self, X, S):
		'''
		Description
			* Predict the class labels of the new classes given their SDs. 
		Input arguments:
			* X (n' x d or n' x n'): 2D np.ndarray - input matrix for test data
			* S (z' x a): 2D np.ndarray - semantic description matrix of test classes
		Return:
			* A 1D np.ndarray of class labels [0, 1, ..., z'-1]
		'''		

		## Assert to call fit first()
		try:
			_ = self.W_
		except AttributeError as exp:
			print('Error! W_ attribute does not exist. Run fit() first. ')
			raise exp
		return np.argmax(self.decision_function(X, S), axis = 1)

	def score(self, X, S, y):
		'''
		Description
			* Compute the accuracy after calling fit()
		Input arguments:
			* X (n' x d or n' x n'): 2D np.ndarray - input matrix for test data
			* S (z' x a): 2D np.ndarray - semantic description matrix of test classes
			* y (n x 1): 1D np.ndarray of class label indices.
		Return:
			* A scalar value. Higher values indicate superior performances. 
		'''			

		## Assert to call fit first()
		try:
			_ = self.W_
		except AttributeError as exp:
			print('Error! W_ attribute does not exist. Run fit() first. ')
			raise exp

		y_pred = self.predict(X, S)
		return accuracy_score(y, y_pred)

def main():

	### To test on gestures ###
	# from zsl_utils.datasets import gestures
	# print('Gesture Data ... ')
	# data_path = r'./data/gesture/data_0.61305.mat'
	# base_dir = dirname(data_path)
	# classes = ['A', 'B', 'C', 'D', 'E']
	# data = gestures.get_data(data_path, debug = True)
	# normalize = False
	# cut_ratio = 1
	# parameters = {'cs__clamp': [3.], # [4., 6., 10.]
	# 			  'fp__skewedness': [6.], # [4., 6., 10.]
	# 			  'fp__n_components': [50],
	# 			  'svm__C': [1.]} # [1., 10.]
	# p_type = 'binary'
	# out_fname = 'dap_gestures.pickle'
	###########################

	###### To test on awa #######
	## This is to convert awa data to a compatible format.
	# from zsl_utils.datasets import awa
	# print('AwA data ...')
	# base_dir = './data/awa'
	# classes = np.loadtxt(join(base_dir, 'testclasses.txt'), dtype = str).tolist()
	# data = awa.get_data(base_dir, debug = True)
	# normalize = False
	# cut_ratio = 1
	# parameters = None
	# p_type = 'binary'
	# out_fname = 'dap_awa.pickle'
	#############################

	###### To test on sun #######
	# from zsl_utils.datasets import sun
	# print('SUN data ...')
	# base_dir = './data/matsun'
	# data = sun.get_data(base_dir, debug = True)
	# classes = [str(idx) for idx in range(data['unseen_attr_mat'].shape[0])]
	# normalize = False
	# cut_ratio = 1
	# parameters = None
	# p_type = 'binary2'
	# out_fname = 'dap_sun.pickle'
	# #############################

	###### To test on awa sae data #######
	## This is to convert awa data to a compatible format.
	from zsl_utils.datasets import awa_sae
	print('AwA data ...')
	base_dir = './data/awa_sae'
	data = awa_sae.get_data(join(base_dir, 'awa_demo_data.mat'), debug = True)
	classes = [str(idx) for idx in range(data['unseen_attr_mat'].shape[0])]
	normalize = False
	cut_ratio = 1
	parameters = None
	p_type = 'binary'
	out_fname = 'dap_awa_sae.pickle'
	#############################

	X_tr, y_tr = data['seen_data_input'], data['seen_data_output']
	X_ts, y_ts = data['unseen_data_input'], data['unseen_data_output']
	S_tr, S_ts = data['seen_attr_mat'], data['unseen_attr_mat']

	clf = SAE()
	clf.fit(X_tr, S_tr, y_tr)
	print('Predicting on train data')
	print(clf.score(X_tr, S_tr, y_tr))

	print('Predicting')
	print(clf.score(X_ts, S_ts, y_ts))

if __name__ == '__main__':
	main()
