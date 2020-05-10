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
from os.path import join
import pickle
from time import time

from copy import deepcopy

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score

from utils import *
from SVMClassifier import *

class DAP(BaseEstimator):
	def __init__(self, skewedness=3., n_components=85, C=100, clamp = 3., rs = None):
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

		Order in which GridSearchCV calls functions in scikit-learn:
			set_params() ==> fit() ==> score()
		'''

		## Parameters
		self.skewedness = skewedness
		self.n_components = n_components
		self.C = C
		self.rs = rs
		self.clamp = clamp

		# Attributes: Modified by fit()
		# TODO:
	
	def _update_attributes(self, X, S, y):
		'''
		Update the following attributes:
			* classes_ (list of class indices)
			* num_classes_ (no. of seen classes)
			* S_ (z x a - seen semantic description matrix)
			* num_attr_ (no. of attributes)
			* feature_size_ (dimension of the data)
		Input arguments:
			* X (n x d - input data matrix)
			* S (z x a - semantic description matrix)
			* y (n x 1 - original class ids)		
		'''
		 # Create data specific attributes.
		self.classes_ = np.unique(y) 
		self.num_classes_ = len(self.classes_)
		assert S.shape[0] == self.num_classes_, 'Error! No. of classes in S and y are inconsistent.'
		self.S_ = S
		self.num_attr_ = S.shape[1]
		self.feature_size_ = X.shape[1]
		self.binary_ = is_binary(S)

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
		##TODO: Add assertions here. 

		self._update_attributes(X, S, y)
		self.S_ = S

		self.clfs_ = []
		if(self.binary_):
			for _ in range(self.num_attr_): self.clfs_.append(SVMClassifier(
				skewedness=self.skewedness, n_components=self.n_components, C=self.C, clamp = self.clamp, rs = self.rs))
		else: 
			for _ in range(self.num_attr_): self.clfs_.append(SVMRegressor(
				skewedness=self.skewedness, n_components=self.n_components, C=self.C, clamp = self.clamp, rs = self.rs))

		Xplat_train, Xplat_val, aplat_train, aplat_val = train_test_split(\
			X, S[y, :], test_size=0.10, random_state = self.rs)			

		for idx in range(self.num_attr_):
			print ('--------- Attribute %d/%d ---------' % (idx+1, self.num_attr_))
			t0 = time()

			# Training and do hyper-parameter search
			self.clfs_[idx].fit(Xplat_train, aplat_train[:,idx])
			print ('Fitted classifier in: %fs' % (time() - t0))
			a_pred_train = self.clfs_[idx].predict(Xplat_train)
			if(self.binary_):
				## Training data
				self.clfs_[idx].set_platt_params(Xplat_val, aplat_val[:,idx])
				f1_score_c0 = f1_score(aplat_train[:, idx], a_pred_train, pos_label = 0)
				f1_score_c1 = f1_score(aplat_train[:, idx], a_pred_train, pos_label = 1)
				print('Train F1 scores: %.02f, %.02f'%(f1_score_c0, f1_score_c1))			
		
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

		## Assertions to call fit first()
		try:
			_ = self.clfs_
		except AttributeError as exp:
			print('Error! clfs_ attribute does not exist. Run fit() first. ')
			raise exp

		a_pred = np.zeros((X.shape[0], S.shape[1]))
		a_proba = np.copy(a_pred)
		prob=[] # (n, 10)

		for idx in range(self.num_attr_):
			a_pred[:,idx] = self.clfs_[idx].predict(X)
			if(self.binary_):
				a_proba[:,idx] = self.clfs_[idx].predict_proba(X)

		if(self.binary_):
			P = a_proba # (n, 85)
			prior = np.mean(self.S_, axis=0)
			prior[prior==0.] = 0.5
			prior[prior==1.] = 0.5    # disallow degenerated priors
			for p in P:
				prob.append(np.prod(S*p + (1-S)*(1-p),axis=1)/\
							np.prod(S*prior+(1-S)*(1-prior), axis=1) )			
		else:
			P = a_pred # (n, 85)
			Md = np.copy(S).astype(np.float)
			Md /= np.linalg.norm(Md, axis = 1, keepdims = True)
			for p in P:
				p /= np.linalg.norm(p)
				prob.append(np.dot(Md, p))

		return prob
	
	def predict(self, X, S):
		'''
		Description
			* Predict the class labels of the new classes given their SDs. 
		Input arguments:
			* X (n' x d or n' x n'): 2D np.ndarray - input matrix for test data
			* S (z' x a): 2D np.ndarray - semantic description matrix of test classes
		Return:
			* A 1D np.ndarray of class labels [0, 1, ..., z']
		'''		
		## Assert to call fit first()
		try:
			_ = self.clfs_
		except AttributeError as exp:
			print('Error! clfs_ attribute does not exist. Run fit() first. ')
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
			_ = self.clfs_
		except AttributeError as exp:
			print('Error! clfs_ attribute does not exist. Run fit() first. ')
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
	

	# TODO: How is this data generated. 
	## AwA
	# data_dir = './awa_data'
	# with open(join(data_dir, 'full_data.pickle'), 'rb') as fp:
	# 	data = pickle.load(fp)['data']
	# cut_ratio = 4

	## SUN
	# from utils import *
	# data = sun_to_dstruct(base_dir = "./matsun")
	# cut_ratio = 1

	### To test on gestures ###
	data_path = r'/home/isat-deep/Desktop/Naveen/fg2020/data/cust_feat_data/data_0.61305.mat'
	classes = ['A', 'B', 'C', 'D', 'E']
	base_dir = './gesture_data'
	data = reformat_dstruct(data_path)
	# normalize = False
	cut_ratio = 1
	# parameters = {'cs__clamp': [3.], # [4., 6., 10.]
	# 			  'fp__skewedness': [6.], # [4., 6., 10.]
	# 			  'fp__n_components': [50],
	# 			  'svm__C': [1.]} # [1., 10.]
	# p_type = 'binary'
	# out_fname = 'dap_' + basename(data_path)[:-4] + '.pickle'
	# print('Gesture Data ... ', p_type)
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

	print('X_tr: ', X_tr.shape)
	print('Y_tr: ', Y_tr.shape)

	X_ts, Y_ts = data['unseen_data_input'], data['unseen_data_output']
	print('X_ts: ', X_ts.shape)
	print('Y_ts: ', Y_ts.shape)

	S_tr, S_ts = data['seen_attr_mat'], data['unseen_attr_mat']
	print('S_tr: ', S_tr.shape)
	print('S_ts: ', S_ts.shape)

	print('Data Loaded. ')
	clf = DAP(skewedness=6., n_components=50, C=1. ,clamp = 3.1, rs = 1)

	print('Fitting')
	clf.fit(X_tr, S_tr, Y_tr)

	print('Predicting on train data')
	print(clf.score(X_tr, S_tr, Y_tr))

	print('Predicting')
	print(clf.score(X_ts, S_ts, Y_ts))
