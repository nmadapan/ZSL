import sys
from os.path import join
import pickle
import numpy as np

# from eszsl import *
from ESZSLClassifier import ESZSL

# ('K: ', (14140, 14140))
# ('Ktt: ', (200, 14140))
# ('Y: ', (14140, 707))
# ('Str: ', (707, 102))
# ('Stt: ', (10, 102))

awa_dir = './gesture_data'
awa_data_path = join(awa_dir, 'full_data.pickle')

with open(awa_data_path, 'rb') as fp:
	data = pickle.load(fp)['data']

X_tr = data['seen_data_input']
Y_tr = data['seen_data_output']

class_ids = np.unique(Y_tr)

cut_ratio = 4

new_y_tr = []
for idx in class_ids:
	temp = np.nonzero(Y_tr == idx)[0]
	last_id = int(len(temp)/cut_ratio)
	new_y_tr += temp[:last_id].tolist()
new_y_tr = np.array(new_y_tr)

Y_tr = Y_tr[new_y_tr]
X_tr = X_tr[new_y_tr, :]

print('X_tr: ', X_tr.shape)
print('Y_tr: ', Y_tr.shape)

X_ts = data['unseen_data_input']
Y_ts = data['unseen_data_output']

print('X_ts: ', X_ts.shape)
print('Y_ts: ', Y_ts.shape)


S_tr = data['seen_attr_mat']
S_ts = data['unseen_attr_mat']

lambdap = 1e-2
sigmap = 1e1

print('Data Loaded. ')
clf = ESZSL(sigmap = sigmap, lambdap = lambdap, degree = 1)

print('Fitting')
clf.fit(X_tr, S_tr, Y_tr)

print('Predicting on train data')
Z = clf.predict(X_tr, S=S_tr)
print(np.mean(Z==Y_tr))

print('Predicting')
Z = clf.predict(X_ts, S=S_ts)
print(np.mean(Z==Y_ts))

