import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler

#####################################################
#################### Transformer ####################
#####################################################

class CustomScaler(StandardScaler):
	def __init__(self, clamp = 3.0):
		super().__init__()
		self.clamp = clamp

	def transform(self, X):
		X = np.copy(X) - self.mean_
		X /= self.scale_
		X[X > self.clamp] = self.clamp
		X[X < -1 * self.clamp] = -1 * self.clamp
		return X

#####################################################
####################### SUN Data ####################
#####################################################

def sun_to_dstruct(base_dir = "./matsun"):

	kf = os.path.join(base_dir,"kernel.mat")
	expidxf = os.path.join(base_dir,"experimentIndices.mat")
	sf = os.path.join(base_dir,"attrClasses.mat")

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

	return data

######################################################
######### Plotting confusion and roc curves ##########
######################################################

def plot_confusion(confusion, classes, wpath = ''):
	fig=plt.figure()
	plt.imshow(confusion,interpolation='nearest',origin='upper')
	plt.clim(0,1)
	plt.xticks(np.arange(0,len(classes)),[c.replace('+',' ') for c in classes],rotation='vertical',fontsize=24)
	plt.yticks(np.arange(0,len(classes)),[c.replace('+',' ') for c in classes],fontsize=24)
	plt.axis([-.5, len(classes)-.5, -.5, len(classes)-.5])
	plt.setp(plt.gca().xaxis.get_major_ticks(), pad=18)
	plt.setp(plt.gca().yaxis.get_major_ticks(), pad=12)
	fig.subplots_adjust(left=0.30)
	fig.subplots_adjust(top=0.98)
	fig.subplots_adjust(right=0.98)
	fig.subplots_adjust(bottom=0.22)
	plt.gray()
	plt.colorbar(shrink=0.79)
	if(len(wpath) == 0):
		plt.show()
	else:
		plt.savefig(wpath)
	return 

def plot_roc(P, GT, classes, wpath = ''):
	AUC=[]
	CURVE=[]
	for i,c in enumerate(classes):
		fp, tp, _ = roc_curve(GT == i,  P[:,i])
		roc_auc = auc(fp, tp)
		print ("AUC: %s %5.3f" % (c,roc_auc))
		AUC.append(roc_auc)
		CURVE.append(np.array([fp,tp]))

	print ("----------------------------------")
	print ("Mean classAUC %g" % (np.mean(AUC)*100))

	order = np.argsort(AUC)[::-1]
	styles=['-','-','-','-','-','-','-','--','--','--']
	plt.figure(figsize=(9,5))
	for i in order:
		c = classes[i]
		plt.plot(CURVE[i][0],CURVE[i][1],label='%s (AUC: %3.2f)' % (c,AUC[i]),linewidth=3,linestyle=styles[i])
	
	plt.legend(loc='lower right')
	plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
	plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
	plt.xlabel('false positive rate',fontsize=18)
	plt.ylabel('true positive rate',fontsize=18)
	if(len(wpath) == 0): plt.show()
	else: plt.savefig(wpath)
	return AUC, CURVE

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

def plot_attAUC(P, y_true, wpath, attributes = None):
	AUC=[]
	if(attributes is None):
		attributes = map(str, range(y_true.shape[1]))

	for i in range(y_true.shape[1]):
		fp, tp, _ = roc_curve(y_true[:,i],  P[:,i])
		roc_auc = auc(fp, tp)
		AUC.append(roc_auc)
	print ("Mean attrAUC %g" % (np.nanmean(AUC)) )

	xs = np.arange(y_true.shape[1])
	width = 0.5

	# fig = plt.figure(figsize=(15,5))
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	rects = ax.bar(xs, AUC, width, align='center')
	ax.set_xticks(xs)
	ax.set_xticklabels(attributes,  rotation=90)
	ax.set_ylabel("area under ROC curve")
	autolabel(rects, ax)
	if(len(wpath) == 0): plt.show()
	else: plt.savefig(wpath)
	return AUC

######################################################
################## General functions #################
######################################################

def loadstr(filename,converter=str):
	return [converter(c.strip()) for c in open(filename).readlines()]

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

def is_binary(mat):
	return len(np.unique(mat[:])) == 2

######################################################
####################### Others #######################
######################################################

def reformat_dstruct(data_path):
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

	return data