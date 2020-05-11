'''
This code processes the Animal With Attributes Dataset and creates
the data in the standard format. 

Dataset:
	1. Download the data from
		https://drive.google.com/file/d/1l0UVhhIU-SmtJ9hqk7OVOG9zNga9qt_I/view
	2. Copy paste this file into a directory. Provide this module the 
		path to this file. 

How to run:
	get_data(data_path, debug = False, use_pickle = True, rerun = False)

Author: Naveen Madapana. 
'''

import sys
import os, os.path
from os.path import dirname, join, isfile
import argparse
import csv

import pickle as cPickle
import bz2

import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat

## Custom modules
from .utils import *

## Main function present in dataset folders. 
def get_data(data_path, debug = False, use_pickle = True, rerun = False):
	'''
	Input arguments:
		* data_path: path to the awa_demo_data.mat file containing the data. 
		* debug: if true, print statements are activated. 
		* use_pickle: if true, it will use 'data.pickle' variable
			before re-creating the data. If use_pickle is true, 
			and 'data.pickle' doesn't exist in data_path, then,
			'data.pickle' is created. 
		* rerun: if true, all data is re-crated even if it is already
			created. 
	Return:
		* data: A dictionary with following keys and values:
			keys - value pairs:
			1. seen_class_ids - 1D np.ndarray of seen class ids.
				(#seen_classes, )
			2. unseen_class_ids - 1D np.ndarray of unseen class ids.
				(#unseen_classes, )
			3. seen_data_input - 2D np.ndarray of features of seen classes. 
				(#seen_instances, #features)
			4. seen_data_output - 1D np.ndarray of re-indexed 
				class ids of seen instances. (#seen_instances, )
			4. unseen_data_input - 2D np.ndarray of features of unseen classes. 
				(#unseen_instances, #features)
			5. unseen_data_output - 1D np.ndarray of re-indexed 
				class ids of unseen instances. (#unseen_instances, )
			6. seen_attr_mat - 2D np.ndarray containing semantic description 
				of seen classes. (#seen_classes x #attributes)
			7. unseen_attr_mat - 2D np.ndarray containing semantic description 
				of unseen classes. (#unseen_classes x #attributes)
	'''
	dir_path = dirname(data_path)
	if(use_pickle and isfile(join(dir_path, 'data.pickle'))):
		print('Reading from: ', join(dir_path, 'data.pickle'))
		with open(join(dir_path, 'data.pickle'), 'rb') as fp:
			data = cPickle.load(fp)['data']
		if(debug): print_dstruct(data)
		return data
	
	## Re-organiz the data in the standard format.
	awa = loadmat(data_path)
	train_data = awa['X_tr']
	test_data = awa['X_te']
	train_class_attributes_labels_continuous_allset = awa['S_tr']
	unseen_class_ids = awa['testclasses_id'].flatten() # absolute test class ids.
	test_class_attributes_labels_continuous = awa['S_te_gt']
	data = {}
	## Seen classes
	data['seen_data_input'] = awa['X_tr'] # n x d
	S_tr = awa['S_tr'] # n x a
	data['seen_attr_mat'], data['seen_data_output'] = np.unique(S_tr, axis = 0, return_inverse = True) # z x a, nx1
	data['seen_class_ids'] = np.array(range(0, data['seen_attr_mat'].shape[0]))

	## Unseen classes
	data['unseen_data_input'] = awa['X_te'] # n x d
	data['unseen_attr_mat'] = awa['S_te_gt'] # z x a
	y_ts = awa['test_labels'][:np.newaxis] == awa['testclasses_id'][:np.newaxis].T
	data['unseen_data_output'] = np.argmax(y_ts, axis = 1) # n x 1
	data['unseen_class_ids'] = unseen_class_ids

	if(debug): print_dstruct(data)
	
	## Save the data into a pickle so we do not need to re-run everything again. 
	if(use_pickle):
		with open(join(dir_path, 'data.pickle'), 'wb') as fp:
			cPickle.dump({'data': data}, fp)

	return data