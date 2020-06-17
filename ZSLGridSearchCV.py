'''
	This file implements a model selection routine for zero-shot
	learning (ZSL) classifiers. These ZSL classifiers should 
	adhere to following protocol. 

	The ZSL class should define the following functions:
		1. set_params(**parameters)
			* This function changes the parameters of the class. 
		2. fit(X, S, y)
			* This function takes three mandatory arguments. 
			* Compute model weights (variables ending with '_')
			* Returns self. 
		3. score(X, S, y)
			* This function takes three mandatory arguments. 
			* Computes a score (for e.g. accuracy). Higher scores
				indicate better performances. 
			* Returns the score. 

	This class, ZSLGridSearchCV, finds the best parameters for the 
	ZSL classifier by doing grid search across the given parameters. 

	Notations:
		* n - No. of instances of seen classes
		* d - Dimension of the data
		* a - No. of attributes/ descriptors
		* z - No. of seen classes
		* X (n x d) - Input data matrix
		* K (n x n) - Kernel matrix
		* S (z x a) - SD matrix of seen classes
		* y (n, ): 1D np.ndarray of class label indices.
		* Y (n x z) - One hot output matrix
		* A (n x a) - Weight matrix

	Author: Naveen Madapana
	Updated: 10 May 2020
'''

## General
import sys
from os.path import join, dirname
import pickle
import random

## Numpy and Sklearn
import numpy as np
from sklearn.model_selection import ParameterGrid

class ZSLGridSearchCV():
	def __init__(self, model, param_dict, cv = 5, rs = None):
		'''
		Input parameters:
			* model: An instance of ZSL classifier
			* param_dict: 
				- key: model parameters
				- value: list of possible values to tune
			* cv: number of folds to consider. 
			* rs: random seed
		Attributes:
			* classes_ (list of class indices)
			* num_classes_ (no. of seen classes)
			* S_ (z x a - seen semantic description matrix)
			* num_attr_ (no. of attributes)
			* feature_size_ (dimension of the data)

			* splits_: A list of sublists. Each sublist consists 
				of class ids in a fold. 
			* data_dict_: 
				key: class id
				value: A 2D np.ndarray containing the input data. 

			* best_model_: An instance of the ZSL classifier. 
			* best_params_: Best parameters from given parameters (param_dict)
			* best_score_: Best average score from all 
				possible combination of parameters. 
		'''

		## Input parameters
		self.model = model
		self.param_dict = param_dict
		self.cv = cv
		self.rs = rs

		## Attributes: Updated by a call to update_attributes()
		# self.classes_ = None
		# self.num_classes_ = None
		# self.S_ = None
		# self.num_attr_ = None
		# self.feature_size_ = None

		## Attributes:
		# self.splits_ = None # Updated by call to _create_splits()
		# self.data_dict_ = None # Updated by call to _create_class_dict()

		## Attributes: Updated by a call to fit()
		# self.best_model_ = None
		# self.best_params_ = None
		# self.best_score_ = None

	def _check_param_dict(self):
		'''
		Check if the parameters in the param_dict are present in the model parameters. 
		'''
		model_keys = self.model.get_params().keys()
		for key in self.param_dict:
			assert key in model_keys, 'Invalid parameter: '+key+' not there in the model'

	def _create_class_dict(self, X, y):
		'''
		Description:
			Compute two variables:
			* dt: dict 
				- keys are class ids
				- values are no. of instances per class. 
			* data_dict: dict 
				- keys are class ids 
				- values are instances of that class in np.ndarray format.
		Input arguments:
			* X (n x d) - Input data matrix
			* y (n, ): 1D np.ndarray of class label indices.
		Attributes:
			* data_dict_
		Return:
			* dt
			* data_dict
		'''
		## Find all the class ids. 
		classes = np.unique(y) 
		## Find no. of instances per class 
		dt = {}
		for class_id in classes:
			dt[class_id] = np.sum(y == class_id)
		## Find the instances per class and re-organize it in a dictionary. 
		data_dict = {}
		for class_id in classes:
			data_dict[class_id] = X[y==class_id, :]
		self.data_dict_ = data_dict # ATTRIBUTE
		return dt, data_dict

	def _create_splits(self, X = None, y = None): ## TODO: Remove X and y later
		'''
		Description:
			This function randomizes and splits the class ids into self.cv folds.
			Generates a list of sublists. Each sublist is a fold containing class ids. 
		Input arguments:
			* X (n x d) - Input data matrix
			* y (n, ): 1D np.ndarray of class label indices.
		Attributes:
			* splits_: list of sublists. Each sublist is a fold containing class ids. 
		Return:
			* splits_
		'''
		## Randomize class indices
		class_indices = np.copy(self.classes_).tolist()
		# NOTE: np.random.shuffle is unstable for some reason. 
		random.Random(self.rs).shuffle(class_indices) 
		## Split class indices into folds
		num_classes_per_fold = int(np.floor(self.num_classes_ / self.cv ))
		splits = [class_indices[fold_idx:fold_idx+num_classes_per_fold] \
					for fold_idx in range(0,self.num_classes_,num_classes_per_fold)]
		## Add the leftover classes to the last fold. Handling the border case. 
		if len(splits[-1]) < num_classes_per_fold:
			splits[-2] += (splits[-1])
			del splits[-1]
		self.splits_ = splits # ATTRIBUTE
		return splits

	def _combine_split_into_data(self, split):
		'''
		Description:
			Given a list of integers (class ids), this functions 
			creates data for those class ids. The create data includes:
			* X (n x d - input data matrix)
			* y (n x 1 - original class ids)
			* y_indexed (n x 1 - re-indexed class ids). The class ids in the split
				will the re-indexed based on the order in which they appear. 
				The values in y_indexed range from [0, len(split)-1]
		Input arguments:
			split: list of integers (class ids)
		Return:
			* X (n x d - input data matrix)
			* y (n x 1 - original class ids)
			* y_indexed (n x 1 - re-indexed class ids). The class ids in the split
				will the re-indexed based on the order in which they appear. 
				The values in y_indexed range from [0, len(split)-1]
		'''
		X = None
		y = []
		y_indexed = []
		try:
			_ = self.data_dict_
		except AttributeError as exp:
			print('Error! data_dict_ attribute does not exist. Run _create_class_dict() first. ')
			raise exp
		for idx, split_val in enumerate(split):
			if(X is None): X = self.data_dict_[split_val]
			else: X = np.append(X, self.data_dict_[split_val], axis=0)
			y += [split_val]*self.data_dict_[split_val].shape[0]
			y_indexed += [idx]*self.data_dict_[split_val].shape[0]
		return X, y, y_indexed

	def _create_train_valid_ids(self, split_idx):
		'''
		Description:
			Given a fold index or split index, it creates the training and
			validation ids by using self.splits_ attribute. 
		Input arguments:
			* split_idx: an integer that ranges from [0, len(splits)-1]
		Return:
			* train_class_ids: A 1D np.ndarray containing class ids. 
			* valid_class_ids: A 1D np.ndarray containing class ids. 
		'''
		try:
			_ = self.splits_
		except AttributeError as exp:
			print('Error! splits_ attribute does not exist. Run _create_splits() first. ')
			raise exp
		train_splits = [self.splits_[idx] for idx in range(len(self.splits_)) if idx != split_idx]
		train_class_ids = [item for sublist in train_splits for item in sublist]
		valid_class_ids = self.splits_[split_idx]
		return train_class_ids, valid_class_ids

	def update_attributes(self, X, S, y):
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

	def fit(self, X, S, y):
		'''
		Description:
			Perform cross-validation and find the best model. 
		Attributes:
			* best_model_: An instance of the ZSL classifier. 
			* best_params_: Best parameters from given parameters (param_dict)
			* best_score_: Best average score from all 
				possible combination of parameters. 
		Input arguments:
			* X (n x d - input data matrix)
			* S (z x a - semantic description matrix)
			* y (n x 1 - original class ids)		
		Return:
			self.
		'''
		# Check if given parameters (param_dict) are present in the model. 
		self._check_param_dict() 
		# Creates data specific attributes
		self.update_attributes(X, S, y) 

		# Creates the attribute data_dict_
		self._create_class_dict(X, y) 
		# Creates the attribute _splits
		self._create_splits(X, y) 

		# Create a grid of parameters
		param_grid = list(ParameterGrid(self.param_dict))

		best_params = None
		best_score = -1 * np.inf
		for params in param_grid:
			self.model.set_params(**params)
			score_list = []
			for split_idx in range(self.cv):
				tr_ids, val_ids = self._create_train_valid_ids(split_idx)
				X_tr, _, y_tr_indexed = self._combine_split_into_data(tr_ids)
				X_val, _, y_val_indexed = self._combine_split_into_data(val_ids)
				S_tr, S_val = S[tr_ids, :], S[val_ids, :]
				self.model.fit(X_tr, S_tr, y_tr_indexed)
				score_list.append(self.model.score(X_val, S_val, y_val_indexed))
			if(np.mean(score_list) > best_score):
				best_score = np.mean(score_list)
				best_params = params
				print('Best score: %.02f'%best_score, params)

		self.best_params_ = best_params # Attribute
		self.best_score_ = best_score # Attribute

		self.model.set_params(**best_params)
		self.model.fit(X, S, y)
		self.best_model_ = self.model # Attribute

		return self.model

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

	cut_ratio = 4
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

	X_ts = data['unseen_data_input']
	Y_ts = data['unseen_data_output']

	print('X_ts: ', X_ts.shape)
	print('Y_ts: ', Y_ts.shape)

	S_tr = data['seen_attr_mat']
	S_ts = data['unseen_attr_mat']

	print('Training ... ')

	# from DAP import DAP as NN
	# model = NN(clamp = 3.1)
	# param_dict = {'skewedness': [4., 6., 10.],
	# 			  'n_components': [40, 50],
	# 			  'C': [1., 10.]}

	from IAP import IAP as NN
	model = NN(clamp = 3.1)
	param_dict = {'skewedness': [4., 6., 10.],
				  'n_components': [40, 50],
				  'C': [1., 10.]}

	# from ESZSL import ESZSL as NN
	# model = NN(degree = 1)
	# param_dict = {'sigmap': [1e-2, 1e-1, 1e0, 1e1], 'lambdap': [1e-2, 1e-1, 1e0, 1e1]}

	# from SAE import SAE as NN
	# model = NN()
	# param_dict = {'lambdap': [2e5, 3e5, 4e5, 5e5]}

	clf = ZSLGridSearchCV(model, param_dict)
	best_model = clf.fit(X_tr, S_tr, Y_tr)
	print(clf.best_params_)
	print(best_model.score(X_ts, S_ts, Y_ts))