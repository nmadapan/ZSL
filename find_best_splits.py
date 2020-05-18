import sys
from os.path import dirname, basename, join
import csv

import numpy as np
import matplotlib.pyplot as plt

from zsl_utils import find_best_seen_unseen_splits, print_dstruct

'''
Description: 
	Given the path to the csv file consisting of annotated descriptors, 
	this script finds the best seen/unseen class splits. In other words, 
	we want all the descriptors in the seen classes to be both 
	present and absent (there should be some zeros and some ones) so 
	that the ZSL classifiers can learn to recognize those attributes. 

	In this script, we first identify the descriptors that are present/
	absent only few times (\\approx <4) and appends the corresponding
	classes to the list of seen classes. 

Input arguments:
	* csv_path: Absolute path to the annotated_videos.csv file. 
		This csv file consists of 12117 rows including the row header,
		24 columns. Each row corresponds to an instance in the CGD 2016
		dataset. 
		- Columns: 1 - 2
			* path to RGB and Depth video
		- Columns: 3
			* Absolute class label
		- Columns: 4 - 22
			* Binary annotations of 19 semantic descriptors. 
		- Columns: 23 - 24
			* Path to the VGG 19 features. 
	* display: If True, SD matrix is visualized. 
	* num_unseen_classes: No. of unseen classes
Return:
	* 
'''

def get_cgd2016_data(csv_path, feature_map_dir, num_unseen_classes = 10, display = False):
	## Read the CSV file. 
	sd_data = []
	with open(csv_path, 'r') as fp:
		for line in csv.reader(fp, delimiter = ','):
			sd_data.append(line)
	# sd_data consists of semantic description information for all the instances.
	sd_data = np.array(sd_data)

	# Absolute class labels of all instances # (12116, )
	labels = sd_data[1:, 2].astype(int)
	# SD annotations of all instances # (12116, 19)
	M = sd_data[1:, 3:-2].astype(int)
	# Get path to RGB feature maps. 
	rgb_fp_files = sd_data[1:, -2]
	rgb_fp_files = [basename(fp) for fp in rgb_fp_files]

	## Find SD vector of each class (there are 48 classes)
	unique_labels, unique_indices = np.unique(labels, return_index = True)

	## Absolute class ids of 48 classes
	absolute_class_ids = unique_labels
	## SD matrix of 48 classes
	S = M[unique_indices, :]

	## Find num classes and descriptors
	num_classes = S.shape[0]
	num_descs = S.shape[1]
	# print('No. of classes: ', num_classes)
	# print('No. of descriptors: ', num_descs)

	seen_class_ids, unseen_class_ids = find_best_seen_unseen_splits(S, num_unseen_classes, K = 3)

	## Find seen/unseen SD matrices
	seen_attr_mat = S[seen_class_ids, :]
	unseen_attr_mat = S[unseen_class_ids, :]

	# print('Seen Attribute Matrix: ', seen_attr_mat.shape)
	# print('Unseen Attribute Matrix: ', unseen_attr_mat.shape)

	## Find seen/unseen flags
	seen_flags = np.sum(labels == absolute_class_ids[seen_class_ids][:, np.newaxis], axis = 0) == 1
	unseen_flags = np.sum(labels == absolute_class_ids[unseen_class_ids][:, np.newaxis], axis = 0) == 1
	assert seen_flags.sum() + unseen_flags.sum() == len(labels), \
		'Error! Sum of no. of seen and unseen flags should be equal to total number of instances. '

	if(display):
		plt.imshow(S.T.astype(np.uint8)*255)
		plt.xlabel('Classes: 0 - ' +  str(S.shape[0]), fontsize = 12)
		plt.ylabel('Descriptors 0 - ' + str(S.shape[1]), fontsize = 12)
		plt.title('Semantic Description Matrix', fontsize = 14)
		plt.show()

	def read_npy(fpath):
		# M_* is RGB, K_* is depth file. 
		return np.load(fpath).flatten()

	data = np.zeros((len(labels), 25 * 512))
	for idx, fname in enumerate(rgb_fp_files):
		data[idx, :] = read_npy(join(feature_map_dir, fname))

	seen_data_input = data[seen_flags, :]
	unseen_data_input = data[unseen_flags, :]

	rel_labels = np.argmax(labels == absolute_class_ids[:, np.newaxis], axis = 0)
	seen_data_output = rel_labels[seen_flags]
	unseen_data_output = rel_labels[unseen_flags]

	data = {}
	data['seen_class_ids'] = seen_class_ids
	data['unseen_class_ids'] = unseen_class_ids
	data['absolute_class_ids'] = absolute_class_ids

	data['seen_data_input'] = seen_data_input
	data['unseen_data_input'] = unseen_data_input

	data['seen_data_output'] = seen_data_output
	data['unseen_data_output'] = unseen_data_output

	data['seen_attr_mat'] = seen_attr_mat
	data['unseen_attr_mat'] = unseen_attr_mat

	return data

if __name__ == '__main__':
	####################
	## Initialization ##
	####################
	base_path =  r'/media/isat-deep/AHRQ IV/Naveen/ie590_project/fg2020_ie590/data'
	csv_path = join(base_path, 'annotated_videos.csv')
	feature_map_dir = join(base_path, 'feature_maps')
	num_unseen_classes = 10
	display = False
	####################

	data = get_cgd2016_data(csv_path, feature_map_dir, num_unseen_classes, display)
	print_dstruct(data)