'''
This code will create the Gesture dataset into
the data in the standard format. 

Dataset:
	* Dataset is generated in Windows system (Naveen)
	* Copy paste this file into a directory. Provide this module the 
		path to this file. 

How to run:
	get_data(data_path, debug = False, use_pickle = True, rerun = False)
	* data_path: path pointing to the .mat file created in windows system.
		For instance: '/home/isat-deep/Desktop/Naveen/fg2020/data/cust_feat_data/data_0.61305.mat'

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

from .utils import dstruct_to_standard, print_dstruct

## Main function present in dataset folders. 
def get_data(data_path, debug = False, use_pickle = True, rerun = False):
	'''
	Input arguments:
		* data_path: path to the .mat file containing the data. 
			For instance: '/home/isat-deep/Desktop/Naveen/fg2020/data/cust_feat_data/data_0.61305.mat'
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
	data = dstruct_to_standard(data_path, debug = debug)

	if(debug): print_dstruct(data)
	
	## Save the data into a pickle so we do not need to re-run everything again. 
	if(use_pickle):
		with open(join(dir_path, 'data.pickle'), 'wb') as fp:
			cPickle.dump({'data': data}, fp)

	return data