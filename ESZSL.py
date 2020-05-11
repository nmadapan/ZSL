'''
	This file implements a zero-shot learning (ZSL) classifier 
	known as ESZSL (Embarassingly simple approach to zero-shot
	learning) proposed by Bernardino et al. 

	The class, ESZSL is implemented in a format similar to 
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
		* K (n x n) - Kernel matrix
		* S (z x a) - SD matrix of seen classes
		* Y (n x z) - One hot output matrix
		* A (n x a) - Weight matrix

	Closed form solution using ESZSL:
		* $$ A = (K^T * K + \gamma I)^-1 XYS^T (SS^T + \lambda I)^-1 $$
		* $\gamma$ and $\lambda$ are hyperparameters. 

	Input to Attributes:
		* $$ S' = A^T K^T $$

	Input to classes:
		* $$ Y' = S A^T K^T $$

	Code adapted from: 
		* https://github.com/nmadapan/Embarrassingly-simple-ZSL.git
		* https://github.com/nmadapan/ESZSL.git

	Link to the paper:
		* http://proceedings.mlr.press/v37/romera-paredes15.pdf

	Author: Naveen Madapana
	Updated: 10 May 2020
'''

import sys
from os.path import join, dirname
import pickle

from copy import deepcopy

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.base import BaseEstimator

class ESZSL(BaseEstimator):
	def __init__(self, lambdap = 0.1, sigmap = 0.1, degree = 'precomputed', \
									rs = None, debug = False):
		'''
		Description:
			* This class inherits BaseEstimator which defines 
				get_params() and set_params() functions. 
		Input parameters:
			lambdap: Regularization parameter for kernel/feature space
			sigmap: Regularization parameter for Attribute Space
			degree: If integer value, polynomial kernel with that degree
				is used. If 'precomputed', the input matrix is expected
				to be a square matrix and it is used directly without
				computing the kernel. 
			rs: random seed
			debug: if True, print statements are activated

		Order in which GridSearchCV calls functions in scikit-learn:
			set_params() ==> fit() ==> score()
		'''

		## Parameters
		self.sigmap = sigmap
		self.lambdap = lambdap
		self.degree = degree
		self.rs = rs
		self.debug = debug

		# Attributes: Created by fit()
		# self.A_ = None
		# self.X_ = None # Needed for computing kernel
	
	def _one_hot_matrix(self, y):
		# y - 1D np.ndarray of class indices (integers [0, 1, ...])
		# Convert y to a one hot matrix. 
		I = np.eye(len(np.unique(y)))
		Y = I[y, :]
		## NOTE: If changed to -1, accuracy drops significantly for both 
		# AwA and gestures dataset. Using 0 instead. 
		Y[Y == 0] = 0.0 
		return Y
	
	def fit(self, X, S, y):
		'''
		Input arguments:
			* X (n x d or n x n): 2D np.ndarray - input matrix
			* S (z x a): 2D np.ndarray - semantic description matrix
			* y (n x 1): 1D np.ndarray of class label indices.
		Math:
			* $$ A = (K^T * K + \gamma I)^-1 XYS^T (SS^T + \lambda I)^-1 $$
		Attributes created:
			* X_ (input data is saved if degree is integer for computing
				kernel for testing data.)
			* A_ (weight matrix to transform inputs to SDs)
		Return:
			* self
		'''
		## Assertion on degree. It should be in ['precomputed' or int]
		assert self.degree == 'precomputed' or isinstance(self.degree, float) \
				or isinstance(self.degree, int), 'Error: degree is invalid!'

		if self.degree =='precomputed':
			K = X
			## Assert that kernel should be squared matrix. 
			assert K.shape[0] == K.shape[1], 'If degree is "precomputed", \
										kernel matrix should be square matrix.'
		else:
			self.X_ = X
			## NOTE: if gamma is not equal to 1, accuracy drops significantly
			# for both awa and gesture datasets. 
			K = polynomial_kernel(X, X, degree = self.degree, gamma = 1)
		
		Y = self._one_hot_matrix(y)
		KK = np.dot(K.T,K)
		KK = np.linalg.inv(KK+self.lambdap*(np.eye(K.shape[0])))
		KYS = np.dot(np.dot(K,Y),S)    
		SS = np.dot(S.T,S)
		SS = np.linalg.inv(SS+self.sigmap*np.eye(SS.shape[0]))
		self.A_ = np.dot(np.dot(KK,KYS),SS)
		return self
		
	def decision_function(self, X, S):
		'''
		Input arguments:
			* X (n' x d or n' x n'): 2D np.ndarray - input matrix for test data
			* S (z' x a): 2D np.ndarray - semantic description matrix of test classes
		Math:
			* $$ Y' = S A^T K^T $$
		Return:
			* Z: (n' x z') matrix of class scores
		'''

		## Assert to call fit first()
		try:
			_ = self.A_
		except AttributeError as exp:
			print('Error! A_ attribute does not exist. Run fit() first. ')
			raise exp

		# self.degree is asserted in fit()		
		if self.degree =='precomputed':
			## Assert that kernel should be squared matrix. 
			assert K.shape[0] == K.shape[1], 'If degree is "precomputed", \
										kernel matrix should be square matrix.'		
			K = X
		else:
			## NOTE: if gamma is not equal to 1, accuracy drops significantly
			# for both awa and gesture datasets. 
			K = polynomial_kernel(X, self.X_, degree = self.degree, gamma = 1)
		Z = np.dot(np.dot(S,self.A_.T),K.T).T

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
			_ = self.A_
		except AttributeError as exp:
			print('Error! A_ attribute does not exist. Run fit() first. ')
			raise exp

		return np.argmax(self.decision_function(X, S), axis=1)

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
			_ = self.A_
		except AttributeError as exp:
			print('Error! A_ attribute does not exist. Run fit() first. ')
			raise exp

		y_pred = self.predict(X, S)
		return np.mean(y == y_pred)

	# def set_params(self, **parameters):
	# 	'''
	# 	Description:
	# 		* Set the value of parameters of the class. 
	# 		* Ideally, only the values passed to the __init__ should be modified. 
	# 	Input arguments:
	# 		* **parameters is a dictionary whose keys are variables and values are variables' values. 
	# 	Return:
	# 		* self
	# 	'''
	# 	for parameter, value in parameters.items():
	# 		setattr(self, parameter, value)
	# 	return self

	# def get_params(self):
	# 	'''
	# 	Description:
	# 		* Get the value of parameters of the class. 
	# 		* Only the values passed to the __init__ are returned. 
	# 	Return:
	# 		* dictionary: key - variable names, value - variables' values
	# 	'''
	# 	dt = deepcopy(self.__dict__)
	# 	for key, value in self.__dict__.items():
	# 		if(key.endswith('_')): dt.pop(key)
	# 	return dt
		
if __name__ == '__main__':
	### To test on gestures ###
	from zsl_utils.datasets import gestures
	print('Gesture Data ... ')
	data_path = r'./data/gesture/data_0.61305.mat'
	base_dir = dirname(data_path)
	classes = ['A', 'B', 'C', 'D', 'E']
	data = gestures.get_data(data_path, debug = True)
	normalize = False
	cut_ratio = 1
	parameters = {'cs__clamp': [3.], # [4., 6., 10.]
				  'fp__skewedness': [6.], # [4., 6., 10.]
				  'fp__n_components': [50],
				  'svm__C': [1.]} # [1., 10.]
	p_type = 'binary'
	out_fname = 'dap_gestures.pickle'
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
	#############################

	X_tr, y_tr = data['seen_data_input'], data['seen_data_output']

	## Downsample the data: reduce the no. of instances per class
	new_y_tr = []
	for idx in np.unique(y_tr):
		temp = np.nonzero(y_tr == idx)[0]
		last_id = int(len(temp)/cut_ratio)
		new_y_tr += temp[:last_id].tolist()
	new_y_tr = np.array(new_y_tr)
	y_tr = y_tr[new_y_tr]
	X_tr = X_tr[new_y_tr, :]

	print('X_tr: ', X_tr.shape)
	print('y_tr: ', y_tr.shape)

	X_ts, y_ts = data['unseen_data_input'], data['unseen_data_output']
	print('X_ts: ', X_ts.shape)
	print('y_ts: ', y_ts.shape)

	S_tr, S_ts = data['seen_attr_mat'], data['unseen_attr_mat']
	print('S_tr: ', S_tr.shape)
	print('S_ts: ', S_ts.shape)

	print('Data Loaded. ')
	clf = ESZSL(sigmap = 1e1, lambdap = 1e-2, degree = 1)

	print('Fitting')
	clf.fit(X_tr, S_tr, y_tr)

	print('Predicting on train data')
	Z = clf.predict(X_tr, S=S_tr)
	print(np.mean(Z==y_tr))

	print('Predicting')
	Z = clf.predict(X_ts, S=S_ts)
	print(np.mean(Z==y_ts))
