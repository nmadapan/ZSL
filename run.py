import sys
from os.path import join, dirname, basename
import pickle

from copy import deepcopy

import numpy as np

from DAP import DAP
from ESZSL import ESZSL
from IAP import IAP
from SAE import SAE
from ZSLGridSearchCV import ZSLGridSearchCV

def run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts):
	print(clf.__class__.__name__, clf.model.__class__.__name__)
	best_model = clf.fit(X_tr, S_tr, y_tr)
	best_params = clf.best_params_
	train_acc = best_model.score(X_tr, S_tr, y_tr)
	acc = best_model.score(X_ts, S_ts, y_ts)
	print('Train acc: %.02f'%train_acc)
	print('Test acc: %.02f'%acc)
	print('Time: %.02f secs'%(time()-start))
	return acc


### To test on CGD 2016 - gestures ###
from zsl_utils.datasets import gestures
print('Gesture Data ... ')
base_dir = r'/media/isat-deep/AHRQ IV/Naveen/ie590_project/fg2020_ie590/data/zsl_data/'
classes = ['A', 'B', 'C', 'D', 'E']
normalize = False
cut_ratio = 1
###########################

dap_acc = []
iap_acc = []
eszsl_acc = []
sae_acc = []

for idx in range(5):
	data_path = join(base_dir, 'data_'+str(idx)+'.mat')
	print('Using: ', basename(data_path))
	data = gestures.get_data(data_path, use_pickle = False, debug = True)

	X_tr, y_tr = data['seen_data_input'], data['seen_data_output']
	## Downsample the data: reduce the no. of instances per class
	new_y_tr = []
	for idx in np.unique(y_tr):
		temp = np.nonzero(y_tr == idx)[0]
		last_id = int(len(temp)/cut_ratio)
		new_y_tr += temp[:last_id].tolist()
	new_y_tr = np.array(new_y_tr)
	y_tr = y_tr[new_y_tr]
	X_tr = X_tr[new_y_tr, :]

	X_ts, y_ts = data['unseen_data_input'], data['unseen_data_output']
	S_tr, S_ts = data['seen_attr_mat'], data['unseen_attr_mat']

	print('Run: DAP')
	from DAP import DAP as NN
	model = NN(clamp = 3.1)
	param_dict = {'skewedness': [4., 6.],
				  'n_components': [50, 200],
				  'C': [1., 10., 100.]}	
	clf = ZSLGridSearchCV(model, param_dict)
	acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)	
	dap_acc.append(acc)

	print('Run: IAP')
	from IAP import IAP as NN
	model = NN(clamp = 3.1)
	param_dict = {'skewedness': [4., 6.],
				  'n_components': [50, 200],
				  'C': [1., 10., 100.]}	
	clf = ZSLGridSearchCV(model, param_dict)	
	acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)	
	iap_acc.append(acc)

	print('Run: ESZSL')
	from ESZSL import ESZSL as NN
	model = NN(degree = 1)
	param_dict = {'sigmap': [1e-2, 1e-1, 1e0, 1e1, 1e2], 'lambdap': [1e-2, 1e-1, 1e0, 1e1, 1e2]}
	clf = ZSLGridSearchCV(model, param_dict)	
	acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)	
	eszsl_acc.append(acc)

	print('Run: SAE')
	from SAE import SAE as NN
	model = NN()
	param_dict = {'lambdap': [1e5, 2e5, 3e5, 4e5, 5e5]}
	clf = ZSLGridSearchCV(model, param_dict)	
	acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)	
	sae_acc.append(acc)

	with open('results.pickle', 'wb') as fp:
		pickle.dump({'dap':dap_acc, 'iap':iap_acc, 'eszsl':eszsl_acc, 'sae':sae_acc}, fp)

print('DAP: ', [np.round(val, 2) for val in dap_acc])
print('IAP: ', [np.round(val, 2) for val in iap_acc])
print('ESZSL: ', [np.round(val, 2) for val in eszsl_acc])
print('SAE: ', [np.round(val, 2) for val in sae_acc])

print('DAP: ', np.round(np.mean(dap_acc), 2), np.round(np.std(dap_acc), 2))
print('IAP: ', np.round(np.mean(iap_acc), 2), np.round(np.std(iap_acc), 2))
print('ESZSL: ', np.round(np.mean(eszsl_acc), 2), np.round(np.std(eszsl_acc), 2))
print('SAE: ', np.round(np.mean(sae_acc), 2), np.round(np.std(sae_acc), 2))