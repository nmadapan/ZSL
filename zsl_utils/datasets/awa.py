'''
This code processes the Animal With Attributes Dataset and creates
the data in the standard format. 

Dataset:
	1. Download the data from
		http://www.ist.ac.at/~chl/AwA/AwA-features-vgg19.tar.bz2
	2. Unzip the AwA-features-vgg19.tar.bz2. It will create a directory
		'Features' containing the folder 'vgg19'.
	3. Copy paste the 'vgg19' directory to any folder and give this 
		module the path to the folder containing 'vgg19' directory. 

How to run:
	get_data(data_path, debug = False, use_pickle = True, rerun = False)
	# data_path: path pointing to the directory containing 'vgg19' folder. 

Adapted from: https://github.com/chcorbi/AttributeBasedTransferLearning.git

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

## Custom modules
from .utils import *

def createFeaturesVector(path, feat_shape=4096, debug = False, rerun = False):
	'''
	Description:
		Function to compute feature vector with all features extracted from a VGG19. 
		This function creates 'feat' folder in the directory one step above 'path' folder. 
		In the 'feat' folder, following .pic.bz2 files are added: featuresVGG19_CLASSNAME.pic.bz2
		Since there are 50 classes in AwA dataset, there will be 50 such .pic.bz2 files. 
	Input arguments:
		* path: path to the VGG19 folder.
		* feat_shape: by default it is 4096. If you use any other deep learning model, 
			set this variable to appropriate no. of features. 
		* debug: if True, print statements are activated. 
		* rerun: 
			If True, all the files are processed even they are already processed. 
			Set it to False, so files are not processed again. It will speed up things. 
	Return:
		None. 
	'''
	feat_dir_path = join(dirname(dirname(path)), 'feat/')
	feat_shape = feat_shape
	subfolders = [x[0] for x in os.walk(path)][1:]

	for i,subfolder in enumerate(subfolders):
		animal = basename(subfolder)

		nb_files = len([name for name in os.listdir(subfolder) if isfile(join(subfolder, name))])

		picklefile = join(feat_dir_path, 'featuresVGG19_' + animal + '.pic.bz2')
		if(isfile(picklefile) and (not rerun)):
			if(debug):
				print('Already created: ', picklefile)
			continue

		for j in range(1,nb_files+1):
			img_feat_path = join(subfolder, animal + '_{:04d}'.format(j) + '.txt')
			if j==1:
				features = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
				features = features.reshape(((feat_shape,1)))
			else:
				tmp_features = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
				tmp_features = tmp_features.reshape(((feat_shape,1)))
				features = np.concatenate((features,tmp_features), axis=1)

		try:
			os.stat(feat_dir_path)
		except:
			os.mkdir(feat_dir_path)

		if(debug):
			print ("Pickling ",animal, " features to ",picklefile)
		bzPickle(features, picklefile)

def concatenate_set_features(data_path, set_classes, nameset='train', debug = False, rerun = False):
	'''
	Description:
		Concatenate all selected animals features together. This file assumes
		that there is 'feat' folder in data_path containing 50 .pic.bz2 files. 
		Further, this will create 'CreatedData' folder in data_path where
		it will store extracted features in compiled format. This will create
		the following files when nameset = 'train:
		1. train_featuresVGG19.pic.bz2: All features contenated into 
			an np.ndarray. Each row in the array is a 4096 dim feature vector. 
		2. train_features_index.txt: Each row in this file is in the
			following format: <class_name> <no. of features of that class>
	Input arguments:
		* data_path: path to the folder containing 'vgg19' folder. This path
			should contain the 'feat' folder.
		* set_classes: list of classes. 
		* nameset: either 'train' or 'test'
		* debug: if true, print statements are activated. 
		* rerun: if true, everything is re-created even if it exists. 
	Return:
		None
	'''
	feat_dir_path = join(data_path, 'feat/')
	created_dir_path = join(data_path, 'CreatedData/')

	picklefile = join(created_dir_path, nameset + '_featuresVGG19.pic.bz2')
	indexfile = join(created_dir_path, nameset + '_features_index.txt')
	if((isfile(picklefile) and isfile(indexfile)) and (not rerun)):
		if(debug):
			print('Files already exist: ', picklefile, indexfile)
		return # Files already exist.

	index= []
	for i,animal in enumerate(set_classes):
		if(debug):
			print ("Adding %s..." % animal)
		features_file = join(feat_dir_path, "featuresVGG19_" + animal + ".pic.bz2")
		features = bzUnpickle(features_file).T
		if i==0:
			X = features
		else:
			X = np.concatenate((X,features),axis=0)
		index.append((animal,features.shape[0]))
	X = csr_matrix(X)

	try:
		os.stat(created_dir_path)
	except:
		os.mkdir(created_dir_path)

	bzPickle(X, picklefile)
	bzPickle(index, indexfile)

## Main function present in dataset folders. 
def get_data(data_path, debug = False, use_pickle = True, rerun = False):
	'''
	Input arguments:
		* data_path: path to the folder containing 'vgg19' folder.
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
	if(use_pickle and (not rerun)):
		if(isfile(join(data_path, 'data.pickle'))):
			print('Reading from: ', join(data_path, 'data.pickle'))
			with open(join(data_path, 'data.pickle'), 'rb') as fp:
				data = cPickle.load(fp)['data']
			if(debug): print_dstruct(data)
			return data
	
	## Creates 'feat' folder
	createFeaturesVector(join(data_path, 'vgg19/'), feat_shape=4096, debug = debug, rerun = rerun)
	
	# Training classes
	if(debug): print ('#### Concatenating training data....')
	trainclasses = loadstr(join(data_path, 'trainclasses.txt'))
	concatenate_set_features(data_path, trainclasses, 'train', debug = debug, rerun = rerun)

	# Test classes
	if(debug): print ('#### Concatenating test data....')
	testclasses = loadstr(join(data_path, 'testclasses.txt'))
	concatenate_set_features(data_path, testclasses, 'test', debug = debug, rerun = rerun)

	## Re-organiz the data in the standard format.
	data = awa_to_dstruct(data_path, predicate_type = 'binary', debug = debug)
	
	## Save the data into a pickle so we do not need to re-run everything again. 
	if(use_pickle):
		with open(join(data_path, 'data.pickle'), 'wb') as fp:
			cPickle.dump({'data': data}, fp)

	return data