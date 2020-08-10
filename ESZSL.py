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
from time import time
import pickle

from copy import deepcopy

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
		# self.S_ = None # (SD matrix of seen classes)
	
	def _one_hot_matrix(self, y):
		# y - 1D np.ndarray of class indices (integers [0, 1, ...])
		# Convert y to a one hot matrix. 
		I = np.eye(len(np.unique(y)))
		Y = I[y, :]
		## NOTE: If changed to -1, accuracy drops significantly for both 
		# AwA and gestures dataset. Using 0 instead. 
		Y[Y == 0] = 0.0 
		return Y

	def _normalize(self, M, axis = 0):
		return M / (np.linalg.norm(M, axis = axis, keepdims=True) + 1e-10)
	
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
			* A_ (weight matrix to transform inputs to SDs) - (d x a)
			* S_ (SD matrix of seen classes)
		Return:
			* self
		'''
		## Assertion on degree. It should be in ['precomputed' or int]
		assert self.degree in ['precomputed', None] or type(self.degree) in [int, float], \
			   'Error: degree is invalid!'

		if self.degree =='precomputed':
			## Assert that kernel should be squared matrix. 
			assert X.shape[0] == X.shape[1], \
				'If degree is "precomputed", kernel matrix should be square matrix.'
			K = X
		elif self.degree is None:
			K = X
		else:
			## NOTE: if gamma is not equal to 1, accuracy drops significantly
			# for both awa and gesture datasets. 
			K = polynomial_kernel(X, X, degree = self.degree, gamma = 1)
			self.X_ = X
		
		Y = self._one_hot_matrix(y)
		KK = np.dot(K.T,K)
		KK = np.linalg.inv(KK+self.lambdap*(np.eye(KK.shape[0])))
		KYS = np.dot(np.dot(K.T,Y),S)
		SS = np.dot(S.T,S)
		SS = np.linalg.inv(SS+self.sigmap*np.eye(SS.shape[0]))
		self.A_ = np.dot(np.dot(KK,KYS),SS)
		self.S_ = S
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
		elif self.degree is None:
			K = X
		else:
			## NOTE: if gamma is not equal to 1, accuracy drops significantly
			# for both awa and gesture datasets. 
			K = polynomial_kernel(X, self.X_, degree = self.degree, gamma = 1)
		
		S_pred = np.dot(K, self.A_)

		# Normalize for cosine similarity
		S_pred = self._normalize(S_pred, axis = 1)
		# Normalize each class now # For cosine similarity
		S = self._normalize(S, axis = 1)

		# Class probabilities
		Z = np.dot(S_pred, S.T)

		return Z, S_pred
	
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

		return np.argmax(self.decision_function(X, S)[0], axis=1)

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
		return accuracy_score(y, y_pred)

def make_data(data_path):
	### To test on gestures ###
	from zsl_utils.datasets import gestures
	# print('Gesture Data ... ')
	# data_path = r'./data/gesture/data_0.61305_raw.mat'
	base_dir = dirname(data_path)
	classes = ['A', 'B', 'C', 'D', 'E']
	data = gestures.get_data(data_path, debug = False, use_pickle = False)
	normalize = False
	cut_ratio = 1
	parameters = {'cs__clamp': [3.], # [4., 6., 10.]
				  'fp__skewedness': [6.], # [4., 6., 10.]
				  'fp__n_components': [1],
				  'svm__C': [1.]} # [1., 10.]
	p_type = 'binary'
	out_fname = 'dap_gestures.pickle'
	###########################

	X_tr, Y_tr = data['seen_data_input'], data['seen_data_output']

	## Downsample the data: reduce the no. of instances per class
	new_y_tr = []
	for idx in np.unique(Y_tr):
		temp = np.nonzero(Y_tr == idx)[0]
		last_id = int(len(temp)/cut_ratio)
		new_y_tr += temp[:last_id].tolist()
	new_y_tr = np.array(new_y_tr)
	Y_tr = Y_tr[new_y_tr]
	X_tr = X_tr[new_y_tr, :]

	# print('X_tr: ', X_tr.shape)
	# print('Y_tr: ', Y_tr.shape)

	X_ts, Y_ts = data['unseen_data_input'], data['unseen_data_output']
	# print('X_ts: ', X_ts.shape)
	# print('Y_ts: ', Y_ts.shape)

	S_tr, S_ts = data['seen_attr_mat'], data['unseen_attr_mat']
	# print('S_tr: ', S_tr.shape)
	# print('S_ts: ', S_ts.shape)

	return (X_tr, S_tr, Y_tr), (X_ts, S_ts, Y_ts)

if __name__ == '__main__':
	# ### To test on gestures ###
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
	# ###########################

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
	# cut_ratio = 4
	# parameters = None
	# p_type = 'binary2'
	# out_fname = 'dap_sun.pickle'
	#############################

	###### To test on awa sae data #######
	## This is to convert awa data to a compatible format.
	# from zsl_utils.datasets import awa_sae
	# print('AwA data ...')
	# base_dir = './data/awa_sae'
	# data = awa_sae.get_data(join(base_dir, 'awa_demo_data.mat'), debug = True)
	# classes = [str(idx) for idx in range(data['unseen_attr_mat'].shape[0])]
	# normalize = False
	# cut_ratio = 1
	# parameters = None
	# p_type = 'binary'
	# out_fname = 'dap_awa_sae.pickle'
	#############################

	### To test on CGD 2016 - gestures ###
	# from zsl_utils.datasets import gestures
	# print('Gesture Data ... ')
	# data_path = r'/media/isat-deep/AHRQ IV/Naveen/ie590_project/fg2020_ie590/data/zsl_data/data_3.mat'
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

	# X_tr, y_tr = data['seen_data_input'], data['seen_data_output']
	# ## Downsample the data: reduce the no. of instances per class
	# new_y_tr = []
	# for idx in np.unique(y_tr):
	# 	temp = np.nonzero(y_tr == idx)[0]
	# 	last_id = int(len(temp)/cut_ratio)
	# 	new_y_tr += temp[:last_id].tolist()
	# new_y_tr = np.array(new_y_tr)
	# y_tr = y_tr[new_y_tr]
	# X_tr = X_tr[new_y_tr, :]

	# X_ts, y_ts = data['unseen_data_input'], data['unseen_data_output']
	# S_tr, S_ts = data['seen_attr_mat'], data['unseen_attr_mat']

	# print('Data Loaded. ')

	# from utils import ZSLPipeline, normalize
	# clf = ZSLPipeline([('s', StandardScaler()),
	# 				   ('c', ESZSL(degree = 1, debug = True)),
	# 				  ])
	# clf.set_params(c__sigmap = 1e1, c__lambdap = 1e-2)

	# print('Fitting ...')
	# clf.fit(X_tr, S_tr, y_tr)

	# print('Predicting on train data')
	# # Normalize each attribute first
	# S_tr = normalize(S_tr, axis = 0)
	# print(clf.score(X_tr, S_tr, y_tr))

	# print('Predicting')
	# # Normalize each attribute first
	# S_ts = normalize(S_ts, axis = 0)
	# print(clf.score(X_ts, S_ts, y_ts))

	######################
	######## LOOP ########
	######################

	from utils import ZSLPipeline, normalize
	from glob import glob
	base_dir = r'/home/isat-deep/Desktop/Naveen/fg2020/data/raw_feat_data'
	mat_files = glob(join(base_dir, 'data_*.mat'))

	K = 1
	result = {}
	start = time()
	for mat_fpath in mat_files:
		print(mat_fpath)
		(X_tr, S_tr, Y_tr), (X_ts, S_ts, Y_ts) = make_data(mat_fpath)
		temp = {}
		in_start = time()
		for iter_idx in range(K):
			iter_start = time()
			print('Iteration ', iter_idx, end = ': ')
			clf = ZSLPipeline([('s', StandardScaler()),
							   ('c', ESZSL(degree = 1, debug = True)),
							  ])
			clf.set_params(c__sigmap = 1e1, c__lambdap = 1e-2)
			clf.fit(X_tr, S_tr, Y_tr)
			S_ts = normalize(S_ts, axis = 0)
			acc = clf.score(X_ts, S_ts, Y_ts)
			temp[iter_idx] = [acc]
			print('%.02f'%acc, '%.02f secs'%(time()-iter_start))
		result[mat_fpath] = temp
		print('Time taken %.02f secs\n'%(time()-in_start))

	with open('./results/eszsl_res.pickle', 'wb') as fp:
		pickle.dump({'result': result}, fp)

	print('Total time taken: %.02f secs'%(time()-start))