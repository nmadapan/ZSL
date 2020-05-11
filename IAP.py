'''
	This script has a class (IAP) that facilitates attribute prediction
	using Indirect Attribute Prediction (IAP) approach proposed by 
	Lampert et al. in 2009 and 2014. 

	It is adapted from the GitHub repository of Charles. 
	https://github.com/chcorbi/AttributeBasedTransferLearning.git

	Author: Naveen Madapana
'''

import os, sys
from os.path import isdir, join, basename, dirname
from time import time
import random
import pickle 

# NumPy and plotting
import numpy as np
import matplotlib.pyplot as plt

## Scipy and sklearn
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import roc_curve, auc, f1_score, make_scorer, roc_auc_score, accuracy_score
from sklearn.exceptions import ConvergenceWarning

## Custom modules
from utils import *
from zsl_utils import *
from SVMClassifier import SVMClassifierIAP
from SVMRegressor import SVMRegressorIAP

import warnings
warnings.filterwarnings('ignore')
# GridSearchCV prints warnings irrespective. 

class IAP(object):
	def __init__(self, data_dict, predicate_type = 'binary', rs = None, normalize = True, base_dir = '.'):
		'''
			data_dict:
				A dictionary with the following keys:
					1. seen_data_input: np.ndarray (train_num_instances x feature_size)
					2. unseen_data_input: np.ndarray (test_num_instances x feature_size)
					3. seen_data_output: np.ndarray (train_num_instances, )
					4. unseen_data_output : np.ndarray (test_num_instances, )
					5. seen_class_ids: np.ndarray (num_seen_classes, )
					6. unseen_class_ids: np.ndarray (num_unseen_classes, )
					7. seen_attr_mat: np.ndarray (num_seen_classes, num_attributes)
					8. unseen_attr_mat: np.ndarray (num_unseen_classes, num_attributes)				
			predicate_type:
				A string. For instance, it can be 'binary'. 
				If it is binary, we will use a classifier, otherwise, we will use a regressor. 
			rs:
				An integer indicating the random_state. This will be passed to the 
				functions that has randomness involved. If None, then, there is 
				no random state. 
		'''
		self.rs = rs
		self.predicate_type = predicate_type
		self.binary = (predicate_type == 'binary')
		self.base_dir = base_dir

		## Write the files and results into this directory
		self.write_dir = join(base_dir, 'IAP_' + self.predicate_type)
		try:
			if(not isdir(self.write_dir)): os.makedirs(self.write_dir)
		except Exception as exp: 
			print(exp)

		self.requirements() # Prints data requirements

		## Variables updated  by a call to initialize_data_vars()
		self.seen_data_input, self.seen_data_output = None, None
		self.seen_attr_mat, self.unseen_attr_mat = None, None
		self.seen_class_ids, self.unseen_class_ids = None, None
		self.unseen_data_input, self.unseen_data_output = None, None
		self.num_seen_classes, self.num_unseen_classes = None, None
		self.X_train, self.y_train = None, None
		self.X_test, self.y_test = None, None
		self.clf = None
		self.num_attr = None

		## Variables udpated by a call to preprocess()
		self.seen_mean, self.seen_std = None, None

		## Variables updated by fit(): For unseen data
		self.y_pred = None # Predicted scores of attributes
		self.y_proba = None # Predicted probabilities of attributes

		## Variables updated by a call to evaluate(): All variables are for unseen data
		self.confusion_matrix = None # np.ndarray of shape (num_unseen_classes x num_unseen_classes)
		# probabilities of unseen classes. np.ndarray of shape (num_unseen_classes x num_unseen_classes)
		self.class_prob_matrix = None 
		self.a_proba = None # Attribute probabilities. np.ndarray of shape (num_unseen_samples x num_attr)

		## Variables updated by generate_results
		self.class_auc = None # AUC of each class. 1D np.ndarray of shape (num_unseen_classes, )
		# list of false positive and true positives of each class. List of size (num_unseen_classes, )
		# Each element is a 2D np.ndarray of shape (_, 2). 
		# 1st col - false positives, 2nd col - true positives.
		self.unseen_class_roc_curve = None
		self.unseen_attr_auc = None # AUC of each attribute. 1D np.ndarray of shape (num_attr, )


		self.initialize_data_vars(data_dict)

	def pprint(self):
		print('######################')
		print('### Seen Classes ###')
		print('No. of seen classes: ', self.num_seen_classes)
		print('Seen data input: ', self.seen_data_input.shape)
		print('Seen data output: ', self.seen_data_output.shape)
		print('Seen attribute matrix:', self.seen_attr_mat.shape)
		print('Seen class IDs:', self.seen_class_ids.shape)

		print('### Unseen Classes ###')
		print('No. of unseen classes: ', self.num_unseen_classes)
		print('Uneen data input: ', self.unseen_data_input.shape)
		print('Unseen data output: ', self.unseen_data_output.shape)
		print('Unseen attribute matrix:', self.unseen_attr_mat.shape)
		print('Unseen class IDs:', self.unseen_class_ids.shape)
		print('######################\n')

	def requirements(self):
		print('######################')
		print('data dictionary should contain 8 np.ndarray variables. ')
		print('###### Seen Classes ###')
		print('1. seen_data_input: (#train_samples x #features)')
		print('2. seen_data_output: (#train_samples, )')
		print('3. seen_attr_mat: (#seen_classes, #attributes)')
		print('4. seen_class_ids: (#seen_classes, )')

		print('\n###### Uneen Classes ###')
		print('5. unseen_data_input: (#test_samples x #features)')
		print('6. unseen_data_output: (#test_samples, )')
		print('7. unseen_attr_mat: (#unseen_classes, #attributes)')
		print('8. unseen_class_ids: (#unseen_classes, )')
		print('######################\n')

	def initialize_data_vars(self, data_dict):
		## Seen Unseen data I/O
		self.seen_data_input = data_dict['seen_data_input']
		self.seen_data_output = data_dict['seen_data_output']
		self.unseen_data_input = data_dict['unseen_data_input']
		self.unseen_data_output = data_dict['unseen_data_output']
		
		self.seen_attr_mat = data_dict['seen_attr_mat']
		self.unseen_attr_mat = data_dict['unseen_attr_mat']

		self.seen_class_ids = data_dict['seen_class_ids']
		self.unseen_class_ids = data_dict['unseen_class_ids']
		
		self.num_attr = self.seen_attr_mat.shape[1]
		self.num_seen_classes = self.seen_attr_mat.shape[0]
		self.num_unseen_classes = self.unseen_attr_mat.shape[0]

		## Create training Dataset
		print ('Creating training dataset...')
		self.X_train = self.seen_data_input
		self.y_train = self.seen_data_output

		## Create testing Dataset
		print ('Creating test dataset...')
		self.X_test = self.unseen_data_input
		self.y_test = self.unseen_data_output

		if(normalize):
			self.X_train, self.X_test = self.preprocess(self.X_train, self.X_test, clamp_thresh = 3.0)
		
		# Irrespective of 'binary' true or false, we should do classification first. 
		self.clf = SVMClassifierIAP(rs = self.rs)

		self.pprint()

	def _clear_data_vars(self):
		self.X_train, self.y_train = None, None
		self.X_test, self.y_test = None, None
		self.seen_data_input, self.seen_data_output = None, None
		self.unseen_data_input = None # self.unseen_data_output is not None
		self.y_pred, self.y_proba = None, None

	def save(self, fname):
		self._clear_data_vars()
		with open(join(self.write_dir, fname), 'wb') as fp:
			pickle.dump({'self': self}, fp)

	def preprocess(self, seen_in, unseen_in, clamp_thresh = 3.):
		'''
			Description:
				Does mean normalization and clamping. 
			Inputs:
				seen_in: np.ndarray (num_seen_samples x num_features)
				unseen_in: np.ndarray (num_unseen_samples x num_features)
			Returns
				seen_in: np.ndarray (num_seen_samples x num_features)
				unseen_in: np.ndarray (num_unseen_samples x num_features)
		'''
		seen_mean = np.mean(seen_in, axis = 0)
		seen_std = np.std(seen_in, axis = 0)
		
		## Mean normalization
		seen_in -= seen_mean
		seen_in /= seen_std
		## Clamping
		seen_in[seen_in > clamp_thresh] = clamp_thresh
		seen_in[seen_in < -1 * clamp_thresh] = -1 * clamp_thresh
		
		## Mean normalization
		unseen_in -= seen_mean
		unseen_in /= seen_std
		## Clamping
		unseen_in[unseen_in > clamp_thresh] = clamp_thresh
		unseen_in[unseen_in < -1 * clamp_thresh] = -1 * clamp_thresh

		self.seen_mean = seen_mean
		self.seen_std = seen_std

		return seen_in, unseen_in

	def train(self, model, x_train, y_train, cv_parameters = None):
		# Binary classification with cross validation
		if(cv_parameters is None): 
			model.fit(x_train, y_train)
			return model
		else:
			## When n_jobs = None, it is taking slightly more time to 
			# run than n_jobs = -1 (Equivalent to #processors). However, when n_jobs is -1, 
			# it prints a lot of ConvergenceWarning by sklearn. 

			## For multi-class case, you can not use roc_auc_score as the output of SVC.predict 
			# is inconsistent with y_pred expected by the roc_auc_score
			clf = GridSearchCV(model, cv_parameters, cv = 5, n_jobs = None, \
								scoring = make_scorer(accuracy_score))
			clf.fit(x_train, y_train)
			print(clf.best_params_)
			return clf.best_estimator_

	def fit(self, cv_parameters = None):
		y_pred = np.zeros(self.y_test.shape)
		y_proba = np.zeros((y_pred.shape[0], self.num_seen_classes))

		print('Training model... (takes around 10 min)')
		t0 = time()
		self.clf.clf = self.train(self.clf.clf, self.X_train, self.y_train, cv_parameters)
		print('Training finished in %.02f secs'%(time() - t0))

		## Train evaluation
		y_pred_train = self.clf.predict(self.X_train)
		y_proba_train = self.clf.predict_proba(self.X_train)
		acc = accuracy_score(self.y_train, y_pred_train)
		print('Train Accuracy: %.02f'%acc)

		## Testing evaluation
		y_pred = self.clf.predict(self.X_test)
		y_proba = self.clf.predict_proba(self.X_test)	
		print(y_pred.shape, y_proba.shape)	
		acc = accuracy_score(self.y_test, y_pred)
		print('Test Accuracy: %.02f'%acc)

		print ('Saving files...')
		np.savetxt(join(self.write_dir, 'prediction_SVM'), y_pred)
		np.savetxt(join(self.write_dir, 'probabilities_SVM'), y_proba)
		
		self.y_pred = y_pred
		self.y_proba = y_proba

		return y_pred, y_proba

	def evaluate(self, prob_fpath = None):
		M = self.unseen_attr_mat # (10,85)
		prob=[] # (n, 10)

		if(prob_fpath is None): P = self.y_proba # (n, 85)
		else: P = np.loadtxt(join(self.write_dir, 'probabilities_SVM')) # (n, 85)
		P = np.dot(P, self.seen_attr_mat) # This is attribute probability matrix. 

		if(self.binary):
			prior = np.mean(self.seen_attr_mat, axis=0)
			prior[prior==0.] = 0.5
			prior[prior==1.] = 0.5    # disallow degenerated priors
			for p in P:
				prob.append(np.prod(M*p + (1-M)*(1-p),axis=1)/\
							np.prod(M*prior+(1-M)*(1-prior), axis=1) )			
		else:
			Md = np.copy(M).astype(np.float)
			Md /= np.linalg.norm(Md, axis = 1, keepdims = True)
			for p in P:
				p /= np.linalg.norm(p)
				prob.append(np.dot(Md, p))

		MCpred = np.argmax(prob, axis = 1) # (n, )

		d = self.num_unseen_classes
		confusion=np.zeros([d, d])
		for pl, gt in zip(MCpred, self.unseen_data_output):
			confusion[gt, pl] += 1.

		confusion /= confusion.sum(axis = 1, keepdims = True)

		self.confusion_matrix = confusion
		self.class_prob_matrix = np.asarray(prob)
		self.a_proba = P

		return confusion, np.asarray(prob), self.unseen_data_output

	def generate_results(self, classes):
		'''
			Following instance variables are used to generate results:
			1. self.confusion_matrix
			2. self.class_prob_matrix
			3. self.unseen_attr_mat
			4. self.unseen_data_output
			5. self.a_proba
		'''		
		wpath = join(self.write_dir, 'AwA-ROC-confusion-IAP-'+p_type+'-SVM.pdf')
		plot_confusion(self.confusion_matrix, classes, wpath)

		wpath = join(self.write_dir, 'AwA-ROC-IAP-SVM.pdf')
		plot_roc(self.class_prob_matrix, self.unseen_data_output, classes, wpath)

		wpath = join(self.write_dir, 'AwA-AttAUC-IAP-SVM.pdf')
		a_test = self.unseen_attr_mat[self.unseen_data_output, :]
		plot_attAUC(self.a_proba, a_test, wpath)
		print ("Mean class accuracy %g" % np.mean(np.diag(self.confusion_matrix)*100))

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

	## Downsample the data: reduce the no. of instances per class
	X_tr, Y_tr = data['seen_data_input'], data['seen_data_output']
	new_y_tr = []
	for idx in np.unique(Y_tr):
		temp = np.nonzero(Y_tr == idx)[0]
		last_id = int(len(temp)/cut_ratio)
		new_y_tr += temp[:last_id].tolist()
	new_y_tr = np.array(new_y_tr)
	Y_tr = Y_tr[new_y_tr]
	X_tr = X_tr[new_y_tr, :]
	data['seen_data_input'], data['seen_data_output'] = X_tr, Y_tr

	iap = IAP(data, predicate_type = p_type, normalize = normalize, base_dir = base_dir)
	start = time()
	iap.fit(parameters)
	print('Total time taken: %.02f secs'%(time()-start))
	confusion, prob, L = iap.evaluate()
	iap.generate_results(classes)
	iap.save(out_fname)

	# with open(join(iap.write_dir, 'full_data.pickle'), 'wb') as fp:
	# 	pickle.dump({'data': data}, fp)	


	# with open(join(join(iap.write_dir, 'DAP_'+p_type), out_fname), 'rb') as fp:
	# 	iap = pickle.load(fp)['self']
	# iap.generate_results(classes)