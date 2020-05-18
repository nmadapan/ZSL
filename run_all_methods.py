import sys
from os.path import join, dirname, basename
import pickle
from time import time
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import SkewedChi2Sampler

from DAP import DAP
from ESZSL import ESZSL
from IAP import IAP
from SAE import SAE
from ZSLGridSearchCV import ZSLGridSearchCV
from utils import ZSLPipeline, normalize, CustomScaler

def run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts):
	start = time()
	print(clf.__class__.__name__, clf.model.__class__.__name__)
	best_model = clf.fit(X_tr, S_tr, y_tr)
	best_params = clf.best_params_
	train_acc = best_model.score(X_tr, S_tr, y_tr)
	acc = best_model.score(X_ts, S_ts, y_ts)
	print('\tTrain acc: %.02f'%train_acc)
	print('\tTest acc: %.02f'%acc)
	print('\tTime: %.02f secs'%(time()-start))
	return acc


### To test on gestures ###
from zsl_utils.datasets import gestures
print('Gesture Data ... ')
data_path = r'./data/gesture/data_0.61305.mat'
base_dir = dirname(data_path)
classes = ['A', 'B', 'C', 'D', 'E']
debug = False
cut_ratio = 1
###########################

### To test on CGD 2016 - gestures ###
# from zsl_utils.datasets import gestures
# print('Gesture Data ... ')
# base_dir = r'/media/isat-deep/AHRQ IV/Naveen/ie590_project/fg2020_ie590/data/zsl_data/'
# data_path = join(base_dir, 'data_'+str(idx)+'.mat')
# classes = ['A', 'B', 'C', 'D', 'E']
# normalize = False
# cut_ratio = 1
# debug = False
###########################

dap_acc = []
iap_acc = []
eszsl_acc = []
sae_acc = []

print('Using: ', basename(data_path))
data = gestures.get_data(data_path, use_pickle = False, debug = debug)

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

# print('Run: DAP')
# model = DAP(clamp = 3.1, debug = debug)
# # param_dict = {'skewedness': [4., 6.],
# # 			  'n_components': [50, 200],
# # 			  'C': [1., 10., 100.]}	
# param_dict = {'skewedness': [4.],
# 			  'n_components': [200],
# 			  'C': [100.]}
# clf = ZSLGridSearchCV(model, param_dict)
# acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)
# dap_acc = acc

# print('Run: IAP')
# model = IAP(clamp = 3.1, debug = debug)
# # param_dict = {'skewedness': [4., 6.],
# # 			  'n_components': [50, 200],
# # 			  'C': [1., 10., 100.]}	
# param_dict = {'skewedness': [4.],
# 			  'n_components': [200],
# 			  'C': [100.]}
# clf = ZSLGridSearchCV(model, param_dict)
# acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)
# iap_acc = acc

print('Run: ESZSL')
fp = SkewedChi2Sampler(skewedness=4, n_components=200)
model = ZSLPipeline([('s', CustomScaler(clamp = 3.0)),
					 ('f', fp),
				     ('c', ESZSL(degree = None, debug = debug)),
				    ])
param_dict = {'c__sigmap': [1e-2, 1e-1, 1e0, 1e1, 1e2], 'c__lambdap': [1e-2, 1e-1, 1e0, 1e1, 1e2]}
# param_dict = {'c__sigmap': [1e-1], 'c__lambdap': [1e2]}
clf = ZSLGridSearchCV(model, param_dict)
acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)
eszsl_acc = acc

print('Run: SAE')
fp = SkewedChi2Sampler(skewedness=4, n_components=200)
model = ZSLPipeline([('s', CustomScaler(clamp = 3.0)),
					 ('f', fp),
				     ('c', SAE(degree = None, debug = debug)),
				    ])
param_dict = {'c__lambdap': [1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 1e6]}
# param_dict = {'c__lambdap': [5e5]}
clf = ZSLGridSearchCV(model, param_dict)
acc = run(clf, X_tr, S_tr, y_tr, X_ts, S_ts, y_ts)
sae_acc = acc

with open('results.pickle', 'wb') as fp:
	pickle.dump({'dap':dap_acc, 'iap':iap_acc, 'eszsl':eszsl_acc, 'sae':sae_acc}, fp)

print('##### Results #####')
print('DAP: ', np.round(dap_acc, 2))
print('IAP: ', np.round(iap_acc, 2))
print('ESZSL: ', np.round(eszsl_acc, 2))
print('SAE: ', np.round(sae_acc, 2))
