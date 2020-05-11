'''
This code will create the Scene Understanding Dataset in 
the data in the standard format. 

Dataset:
	* Download the data from
		https://drive.google.com/open?id=1-Y-KbAu_YVz7tXbbYHztFyDXOwrLrcWx
	* Copy paste the files into a directory. Provide this module the 
		path to the directory containing these files. 

How to run:
	get_data(data_path, debug = False, use_pickle = True, rerun = False)
	* data_path: path pointing to the directory containing following
	.mat files:
		1. attrClasses.mat
		2. experimentIndices.mat
		3. kernel.mat
		4. perclassattributes.mat

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

from .utils import sun_to_dstruct, print_dstruct

## Main function present in dataset folders. 
def get_data(data_path, debug = False, use_pickle = False, rerun = False):
	'''
	Input arguments:
		* data_path: path to the folder containing to the files:
			1. attrClasses.mat
			2. experimentIndices.mat
			3. kernel.mat
			4. perclassattributes.mat			
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
	if(use_pickle):
		if(isfile(join(data_path, 'data.pickle'))):
			print('Reading from: ', join(data_path, 'data.pickle'))
			with open(join(data_path, 'data.pickle'), 'rb') as fp:
				data = cPickle.load(fp)['data']
			if(debug): print_dstruct(data)
			return data
	
	## Re-organiz the data in the standard format.
	data = sun_to_dstruct(data_path, debug = debug)

	if(debug): print_dstruct(data)
	
	## Save the data into a pickle so we do not need to re-run everything again. 
	if(use_pickle):
		with open(join(data_path, 'data.pickle'), 'wb') as fp:
			cPickle.dump({'data': data}, fp)

	return data