'''
	This file implements a zero-shot learning (ZSL) classifier 
	known as DAP (Direct Attribute Prediction) proposed by 
	Lampert et al. in 2009 and 2014. 

	The class, DAP is implemented in a format similar to 
	standard scikit learn packages, except that, in addition to
	class labels, you should pass a semantic description (SD)
	matrix as well. 

	NOTE: You CAN NOT use this model directly with other scikit-
	learn packages such as sklearn.model_selection.GridSearchCV
	and sklearn.pipeline.Pipeline in scikit-learn has no good 
	way to pass both class labels and semantic description 
	matrix to methods such as predict(), score() etc. However, 
	you can pass additional arguments (using **kwargs) to
	fit() method. Hence, in this repository, you will find
	modules such as GridSearchCV that are customized to ZSL
	while following the standards of scikit-learn.

	Notations:
		* n - No. of instances of seen classes
		* d - Dimension of the data
		* a - No. of attributes/ descriptors
		* z - No. of seen classes
		* X (n x d) - Input data matrix
		* S (z x a) - SD matrix of seen classes

	In DAP, you will train a binary classifier or regressor for each
	descriptor separately. Hence, there will be $a$ classifiers or
	regressors. Support Vector Machines (SVM) classifier and 
	regressor is used in this file. This implementation expects a
	customized SVM classifier/regressor (refer to SVMClassifier.py
	or SVMRegressor.py).

	Code adapted from: 
		* https://github.com/chcorbi/AttributeBasedTransferLearning.git

	Link to the paper:
		* https://hannes.nickisch.org/papers/articles/lampert13attributes.pdf

	Author: Naveen Madapana
	Updated: 10 May 2020
'''

import sys
from os.path import join, basename, dirname
import pickle
from time import time

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, accuracy_score

from utils import is_binary
from SVMClassifier import SVMClassifier
from SVMRegressor import SVMRegressor

class DAP(BaseEstimator):
	def __init__(self, skewedness=3., n_components=85, C=100, 
							clamp = 2.9, rs = None, debug = False):
		'''
		Description:
			* This class inherits BaseEstimator which defines 
				get_params() and set_params() functions. 
			* Before performing zero shot learning, the Chi2 kernel is estimated
				as suggested in the  original paper. We use the following 
				sklearn module: sklearn.kernel_approximation.SkewedChi2Sampler. 
				This module requires skewedness and n_components as parameters. 
			* Note that the values of initial raw features passed to the 
				SkewedChi2Sampler can not be greater than skewedness value. 
				Hence, the raw data is first normalized and then, clipped using
				clamp on both positive and negative directions. 
		Input parameters:
			skewedness: It is a parameter in the Chi2 kernel. 
			n_components: It is a parameter in the Chi2 kernel. This parameter
				is equivalent to the no. of features that we want out of the
				Chi2 kernel. 
			C: Regularization parameter of the SVM classifier/regressor.
			clamp: This is the maximum absolute value that the normalized features
				are constrained to. If the normalized features exceed clamp, 
				they will be clipped to this value. For instance, if -9 appears 
				in normalized features, it is modified to -2.9 (the value of clamp).
			rs: An integer indicating the random_state. This will be passed to the 
				functions that has randomness involved. If None, then, there is 
				no random state. The modules with randomness are: SkewedChi2Sampler,
				SVC, LinearSVC and train_test_split. 
			debug: if True, print statements are activated. 

		Order in which GridSearchCV calls functions in scikit-learn:
			set_params() ==> fit() ==> score()
		'''

		## Parameters
		self.skewedness = skewedness
		self.n_components = n_components
		self.C = C
		self.rs = rs
		self.clamp = clamp # value of clamp should be less than skewedness
		self.debug = debug

		## Attributes: Modified by fit()
		# self.S_ = None # (z x a - seen semantic description matrix)
		## Input data is saved if degree is integer. 
		## It is used to compute the kernel for testing data.
		# self.X_ = None
		# self.clfs_ = [] # (list of $a$ classifiers/regressors)

		## Attributes: Modified by fit() thorugh _update_attributes()
		# self.classes_ = None # (1D np.ndarray consisting of class indices)
		# self.num_classes_ = None # (no. of seen classes)
		# self.num_attr_ = None # (no. of attributes)
		# self.feature_size_ = None # (dimension of the data)
		# self.binary_	 = None # (True if S is binary, False otherwise.)

		## Attributes: Modified by decision_function()
		# self.class_prob_ = None # class probability matrix.
		# self.attr_prob_ = None # attribute probability matrix.

	def _update_attributes(self, X, S, y):
		'''
		Update the following attributes:
			* classes_ (1D np.ndarray consisting of class indices)
			* num_classes_ (no. of seen classes)
			* num_attr_ (no. of attributes)
			* feature_size_ (dimension of the data)
			* binary_ (True if S is binary, False otherwise.)
		Input arguments:
			* X (n x d - input data matrix)
			* S (z x a - semantic description matrix)
			* y (n x 1 - original class ids)		
		'''
		 # Create data specific attributes.
		self.classes_ = np.unique(y)
		self.num_classes_ = len(self.classes_)
		self.num_attr_ = S.shape[1]
		self.feature_size_ = X.shape[1]
		self.binary_ = is_binary(S)

	def fit(self, X, S, y):
		'''
		Input arguments:
			* X (n x d or n x n): 2D np.ndarray - input matrix
			* S (z x a): 2D np.ndarray - semantic description matrix
			* y (n x 1): 1D np.ndarray of class label indices.
		Attributes created:
			* S_ (z x a - seen semantic description matrix)
			* X_ (input data is saved if degree is integer for computing
				kernel for testing data.)
			* self.clfs_ (list of $a$ classifiers/regressors)
		Return:
			* self
		'''
		## Update data attributes
		self._update_attributes(X, S, y)
		self.S_ = S		

		## Assert - sanity checks
		# The value of clamp can not be greater than skewedness. 
		if(self.clamp > self.skewedness):
			raise ValueError('Error! Value of clamp should be less than skewedness.')
		# No. of classes in S and y should be consistent.
		assert S.shape[0] == self.num_classes_, 'Error! No. of classes in S and y are inconsistent.'
		# The values in y should be relative i.e. they should be 0, 1, ..., num_classes_-1
		if(np.min(y)<0 or np.max(y)>=self.num_classes_):
			raise ValueError('Error! Class indices in y should be relative: 0, 1, ..., num_classes-1')

		## Initialize classifiers/regressors
		self.clfs_ = []
		if(self.binary_):
			for _ in range(self.num_attr_): 
				self.clfs_.append(SVMClassifier(\
				skewedness=self.skewedness, n_components=self.n_components, \
				C=self.C, clamp = self.clamp, rs = self.rs))
		else: 
			for _ in range(self.num_attr_): 
				self.clfs_.append(SVMRegressor(\
				skewedness=self.skewedness, n_components=self.n_components,\
				C=self.C, clamp = self.clamp, rs = self.rs))

		## Training data is used for classification/regression, validation data is used 
		# for platt fitting. 
		Xplat_train, Xplat_val, aplat_train, aplat_val = train_test_split(\
			X, S[y, :], test_size=0.10, random_state = self.rs)			

		for idx in range(self.num_attr_):
			if(self.debug): print ('---- Attribute %d/%d ----' % (idx+1, self.num_attr_))
			t0 = time()

			# Training and do hyper-parameter search
			self.clfs_[idx].fit(Xplat_train, aplat_train[:,idx])
			if(self.debug): print ('Fitted classifier in: %fs' % (time() - t0))
			a_pred_train = self.clfs_[idx].predict(Xplat_train)
			if(self.binary_):
				## Training data
				self.clfs_[idx].set_platt_params(Xplat_val, aplat_val[:,idx])
				f1_score_c0 = f1_score(aplat_train[:, idx], a_pred_train, pos_label = 0)
				f1_score_c1 = f1_score(aplat_train[:, idx], a_pred_train, pos_label = 1)
				if(self.debug): print('Train F1 scores: %.02f, %.02f'%(f1_score_c0, f1_score_c1))			
		
		return self
		
	def decision_function(self, X, S):
		'''
		Input arguments:
			* X (n' x d or n' x n'): 2D np.ndarray - input matrix for test data
			* S (z' x a): 2D np.ndarray - semantic description matrix of test classes
		Attributes:
			* self.class_prob_ (n' x z'): 2D np.ndarray - class probability matrix.
			* self.attr_prob_  (n' x a): 2D np.ndarray - attribute probability matrix.
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

		## Estimate output probabilities. Refer to paper. 
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

		## Attributes
		self.class_prob_ = prob # (n, z)
		self.attr_prob_ = a_proba # (n, a)

		return prob
	
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
		acc = accuracy_score(y, y_pred)
		if(self.debug): print('Accuracy: %.02f '%acc)
		return acc
		
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
