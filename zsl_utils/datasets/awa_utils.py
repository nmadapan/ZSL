import numpy as np
import pickle as cPickle
import bz2
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


def awa_to_dstruct(data_path, predicate_type = 'binary'):

	# Get features index to recover samples
	train_index = bzUnpickle(join(data_path, 'CreatedData/train_features_index.txt'))
	test_index = bzUnpickle(join(data_path, 'CreatedData/test_features_index.txt'))

	# Get classes-attributes relationship
	train_attributes = get_class_attributes(data_path, name='train', predicate_type=predicate_type)
	test_attributes = get_class_attributes(data_path, name='test', predicate_type=predicate_type)

	# Create training Dataset
	print ('Creating training dataset...')
	X_train, y_train = create_data(join(data_path, 'CreatedData/train_featuresVGG19.pic.bz2'),train_index, train_attributes)
	
	# Convert from sparse to dense array
	print ('X_train to dense...')
	X_train = X_train.toarray()

	print ('Creating test dataset...')
	X_test, y_test = create_data(join(data_path, 'CreatedData/test_featuresVGG19.pic.bz2'),test_index, test_attributes)

	# Convert from sparse to dense array
	print ('X_test to dense...')
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