import numpy as np
import pickle as cPickle
import bz2
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.io import loadmat, savemat
from os.path import join, dirname, basename

######################################################
#### Process Animal with Attributes (AwA) Dataset ####
######################################################

'''
Required files
	* classes.txt
	* numexamples.txt
	* testclasses.txt
	* trainclasses.txt
	* predicate-matrix-{binary}.txt # when predicate_type = 'binary'
	* CreatedData/train_features_index.txt
	* CreatedData/test_features_index.txt
	* CreatedData/train_featuresVGG19.pic.bz2
	* CreatedData/test_featuresVGG19.pic.bz2
'''

def nameonly(x):
	return x.split('\t')[1]

def loadstr(filename,converter=str):
	return [converter(c.strip()) for c in open(filename).readlines()]

def loaddict(filename,converter=str):
	D={}
	for line in open(filename).readlines():
		line = line.split()
		D[line[0]] = converter(line[1].strip())
	
	return D

def get_full_animals_dict(path):
	animal_dict = {}
	with open(path) as f:
		for line in f:
			(key, val) = line.split()
			animal_dict[val] = int(key)
	return animal_dict

def get_animal_index(path, filename):
	classes = []
	animal_dict = get_full_animals_dict(join(path, "classes.txt"))
	with open(join(path, filename)) as infile:
		for line in infile:
			classes.append(line[:-1])
	return [animal_dict[animal]-1 for animal in classes]

def get_attributes(data_path):
	attributes = []
	with open(join(data_path, 'attributes.txt')) as infile:
		for line in infile:
			attributes.append(line[:-1])
	return attributes

def get_class_attributes(path, name='train', predicate_type='binary'):
	animal_index = get_animal_index(path, name+'classes.txt')
	classAttributes = np.loadtxt(join(path, "predicate-matrix-" + predicate_type + ".txt"), comments="#", unpack=False)
	return classAttributes[animal_index]

def create_data(path, sample_index, attributes):
  
	X = bzUnpickle(path)
	
	nb_animal_samples = [item[1] for item in sample_index]
	for i,nb_samples in enumerate(nb_animal_samples):
		if i==0:
			y = np.array([attributes[i,:]]*nb_samples)
		else:
			y = np.concatenate((y,np.array([attributes[i,:]]*nb_samples)), axis=0)
	
	return X,y


def autolabel(rects, ax):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		if np.isnan(rect.get_height()):
			continue
		else:
			height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%0.3f' % height, 
			ha='center', va='bottom', rotation=90)


def awa_to_dstruct(data_path, predicate_type = 'binary', debug = False):
	'''
	Description:
		Converts the AwA dataset in the standard format. 
	Input Arguments:
		* base_dir: path pointing to the directory containing following files:
			1. train_featuresVGG19.pic.bz2
			2. train_features_index.txt
			3. test_featuresVGG19.pic.bz2
			4. test_features_index.txt
		* predicate_type: type of attributes.
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

	# Get features index to recover samples
	train_index = bzUnpickle(join(data_path, 'CreatedData/train_features_index.txt'))
	test_index = bzUnpickle(join(data_path, 'CreatedData/test_features_index.txt'))

	# Get classes-attributes relationship
	train_attributes = get_class_attributes(data_path, name='train', predicate_type=predicate_type)
	test_attributes = get_class_attributes(data_path, name='test', predicate_type=predicate_type)

	# Create training Dataset
	if(debug): print ('Creating training dataset...')
	X_train, y_train = create_data(join(data_path, 'CreatedData/train_featuresVGG19.pic.bz2'),train_index, train_attributes)
	
	# Convert from sparse to dense array
	if(debug): print ('X_train to dense...')
	X_train = X_train.toarray()

	if(debug): print ('Creating test dataset...')
	X_test, y_test = create_data(join(data_path, 'CreatedData/test_featuresVGG19.pic.bz2'),test_index, test_attributes)

	# Convert from sparse to dense array
	if(debug): print ('X_test to dense...')
	X_test = X_test.toarray()    

	classnames = loadstr(join(data_path, 'classes.txt'), nameonly)
	numexamples = loaddict(join(data_path, 'numexamples.txt'), int)
	test_classnames=loadstr(join(data_path, 'testclasses.txt'))
	train_classnames=loadstr(join(data_path, 'trainclasses.txt'))

	test_classes = [ classnames.index(c) for c in test_classnames]
	train_classes = [ classnames.index(c) for c in train_classnames]

	test_output = []
	for idx, c in enumerate(test_classes):
		test_output.extend( [idx]*numexamples[classnames[c]] )

	train_output = []
	for idx, c in enumerate(train_classes):
		train_output.extend( [idx]*numexamples[classnames[c]] )

	data = {}

	data['seen_class_ids'] = np.array(train_classes).astype(np.uint8)
	data['unseen_class_ids'] = np.array(test_classes).astype(np.uint8)

	data['seen_data_input'] = X_train
	data['seen_data_output'] = np.array(train_output).astype(np.uint8)
	
	data['unseen_data_input'] = X_test
	data['unseen_data_output'] = np.array(test_output).astype(np.uint8)
	
	data['seen_attr_mat'] = train_attributes
	data['unseen_attr_mat'] = test_attributes

	if(debug): print_dstruct(data)

	return data

######################################################
################## General functions #################
######################################################

def bzPickle(obj,filename):
	f = bz2.BZ2File(filename, 'wb')
	cPickle.dump(obj, f)
	f.close()

def bzUnpickle(filename):
	return cPickle.load(bz2.BZ2File(filename))

def print_dict(dict_inst, idx = 1):
	for key, value in dict_inst.items():
		if(isinstance(value, dict)):
			print('\t'*(idx-1), key, ': ')
			print_dict(value, idx = idx+1)
		else:
			print('\t'*idx, key, ': ', end = '')
			if(isinstance(value, np.ndarray)):
				print(value.shape)
			else: print(value)

def print_dstruct(data):
	print('######################')
	print('### Seen Classes ###')
	print('Seen data input: ', data['seen_data_input'].shape)
	print('Seen data output: ', data['seen_data_output'].shape)
	print('Seen attribute matrix:', data['seen_attr_mat'].shape)
	print('Seen class IDs:', data['seen_class_ids'].shape)

	print('### Unseen Classes ###')
	print('Uneen data input: ', data['unseen_data_input'].shape)
	print('Unseen data output: ', data['unseen_data_output'].shape)
	print('Unseen attribute matrix:', data['unseen_attr_mat'].shape)
	print('Unseen class IDs:', data['unseen_class_ids'].shape)
	print('######################\n')

#####################################################
####################### SUN Data ####################
#####################################################

def sun_to_dstruct(base_dir, debug = False):
	'''
	Description:
		Converts the SUN dataset in the standard format. 
	Input Arguments:
		* base_dir: path pointing to the directory containing following	.mat files:
			1. attrClasses.mat
			2. experimentIndices.mat
			3. kernel.mat
			4. perclassattributes.mat
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

	kf = join(base_dir,"kernel.mat")
	expidxf = join(base_dir,"experimentIndices.mat")
	sf = join(base_dir,"attrClasses.mat")

	expidx = loadmat(expidxf)
	tridx = expidx['trainInstancesIndices'].flatten()-1
	Ytr = expidx['trainInstancesLabels'].flatten()-1
	Citr = expidx['trainClassesIndices'].flatten()-1
	ttidx = expidx['testInstancesIndices'].flatten()-1
	Ytt = expidx['testInstancesLabels'].flatten()-1
	Citt = expidx['testClassesIndices'].flatten()-1

	S = loadmat(sf)['attrClasses']
	Str = S[Citr,:]
	Stt = S[Citt,:]

	K = loadmat(kf)['K']
	Ktt = K[ttidx][:,tridx]
	K = K[tridx][:,tridx]

	## Reformatting the data
	data = {}
	data['seen_class_ids'] = Citr
	data['unseen_class_ids'] = Citt
	data['seen_data_input'] = K
	data['seen_data_output'] = Ytr
	data['unseen_data_input'] = Ktt
	data['unseen_data_output'] = Ytt
	data['seen_attr_mat'] = Str
	data['unseen_attr_mat'] = Stt

	if(debug): print_dstruct(data)

	return data

######################################################
####################### Others #######################
######################################################

def dstruct_to_standard(data_path, debug = False):
	'''
		Description:
			Convert the ZSL data file that we have in Windows/Matlab into a
			format compatible with python DAP codes. 
		Input:
			data_path: path to the .mat file. This file has a matlab struct variable 'dstruct'
				Example: # r'/home/isat-deep/Desktop/Naveen/fg2020/data/raw_feat_data/data_0.11935.mat'
		Output: 
			data: dictionary with the following keys
				1. seen_data_input: np.ndarray (train_num_instances x feature_size)
				2. unseen_data_input: np.ndarray (test_num_instances x feature_size)
				3. seen_data_output: np.ndarray (train_num_instances, )
				4. unseen_data_output : np.ndarray (test_num_instances, )
				5. seen_class_ids: np.ndarray (num_seen_classes, )
				6. unseen_class_ids: np.ndarray (num_unseen_classes, )
				7. seen_attr_mat: np.ndarray (num_seen_classes, num_attributes)
				8. unseen_attr_mat: np.ndarray (num_unseen_classes, num_attributes)

	'''
	x = loadmat(data_path, struct_as_record = False, squeeze_me = True)['dstruct']

	imp_keys = ['unseen_class_ids', 'seen_class_ids', 'seen_data_input', 'unseen_data_input', \
				'seen_data_output', 'unseen_data_output', 'seen_attr_mat', 'unseen_attr_mat']

	data = {}
	for key in imp_keys:
		data[key] = getattr(x, key)
	del x

	data['seen_data_input'] = data['seen_data_input'].astype(np.float)
	data['unseen_data_input'] = data['unseen_data_input'].astype(np.float)

	data['seen_data_output'] = data['seen_data_output'].astype(np.uint8) - 1 # Matlab indices start from 1
	data['unseen_data_output'] = data['unseen_data_output'].astype(np.uint8) - 1 # Matlab indices start from 1
	
	data['seen_class_ids'] = data['seen_class_ids'].astype(np.uint8) - 1 # Matlab indices start from 1
	data['unseen_class_ids'] = data['unseen_class_ids'].astype(np.uint8) - 1 # Matlab indices start from 1

	if(debug): print_dstruct(data)

	return data

def get_cgd2016_data(csv_path, feature_map_dir, num_unseen_classes = 10, display = False):
	'''
	Description: 
		Given the path to the csv file consisting of annotated descriptors, 
		this script finds the best seen/unseen class splits. In other words, 
		we want all the descriptors in the seen classes to be both 
		present and absent (there should be some zeros and some ones) so 
		that the ZSL classifiers can learn to recognize those attributes. 
	
		In this script, we first identify the descriptors that are present/
		absent only few times (\\approx <4) and we append the corresponding
		classes to the list of seen classes. 
	
	Input arguments:
		* csv_path: Absolute path to the annotated_videos.csv file. 
			This csv file consists of 12117 rows including the row header,
			and 24 columns. Each row corresponds to an instance in the CGD
			2016 dataset for 48 gesture categories. 
			- Columns: 1 - 2
				* path to RGB and Depth video
			- Columns: 3
				* Absolute class label
			- Columns: 4 - 22
				* Binary annotations of 19 semantic descriptors. 
			- Columns: 23 - 24
				* Path to the VGG 19 features of RGB and Depth videos. 
		* display: If True, SD matrix is visualized. 
		* num_unseen_classes: No. of unseen classes
	Return:
		data: zsl data in standard format. 
	'''	
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

	## Find a potential seen-unseen class split
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

######################################################
################## Find best splits ##################
######################################################

def find_best_seen_unseen_splits(S, num_unseen_classes, K = 3):
	'''
	Description:
		Given the SD matrix (S - num_classes x num_descriptors), 
		the goal is find a set of seen and unseen class ids so that
		the data imbalance in the seen dataset is minimized. 
		In other words, it is ensured that all descriptors are both
		present and absent in the seen data. 
	Input arguments:
		* S: semantic description matrix of all classes (num_classes, num_descriptors)
		* num_unseen_classes: int. No. of unseen classes. 
		* K: A hyperparameter. 100/K percentage of classes will be first
			added to the list of seen classes. These classes contribute to the most
			data imbalance at attribute level. 
	'''
	assert num_unseen_classes < S.shape[0], \
		'No. of unseen classes cant be greater than total no. of classes. '
	## Find num classes and descriptors
	num_classes = S.shape[0]
	num_descs = S.shape[1]
	## Adaptive threshold. The init_num_seen_classes number of classes
	# that are contributing towards to data imbalance will be added to 
	# the seen classes. Rest of the classes are randomized. 
	init_num_seen_classes = num_classes // K

	freq = S.sum(axis = 0)
	descs_order = np.argsort(freq)
	seen_class_ids = []
	unseen_class_ids = []

	for desc_id in descs_order:
		sd = S[:, desc_id]
		zero_cls_ids = np.nonzero(sd == 0)[0]
		one_cls_ids = np.nonzero(sd == 1)[0]
		if(len(zero_cls_ids) < len(one_cls_ids)):
			seen_class_ids += zero_cls_ids.tolist()
		else:
			seen_class_ids += one_cls_ids.tolist()
		## If minimum of seen_class_ids is reached, then exit. 
		if(len(np.unique(seen_class_ids)) > init_num_seen_classes):
			seen_class_ids = seen_class_ids[0:init_num_seen_classes]
			seen_class_ids = np.unique(seen_class_ids).tolist()
			break
	seen_class_ids = np.unique(seen_class_ids)
	rest_ids = list(set(range(S.shape[0])).difference(seen_class_ids))
	np.random.shuffle(rest_ids)

	## Find seen/unseen class ids (relative)
	unseen_class_ids = np.sort(rest_ids[:num_unseen_classes])
	seen_class_ids = np.sort(np.append(seen_class_ids, rest_ids[num_unseen_classes:]))

	return seen_class_ids, unseen_class_ids