import numpy as np
import scipy
import scipy.io
from scipy.linalg import solve_sylvester
import argparse
import sys
from sklearn.base import BaseEstimator

class SAE(BaseEstimator):
	def __init__(self, lambdap = 5e5, rs = None):
		self.lambdap = lambdap
		self.rs = rs

	def _normalize(self, M, axis = 0):
		return M / (np.linalg.norm(M, axis = axis, keepdims=True) + 1e-10)

	def fit(self, X, S, y):
		## Assertions here
		X = self._normalize(X)

		# S: z x a ==> n x a # X: n x d
		A = np.dot(S[y, :].T, S[y, :]) # a x a
		B = self.lambdap * np.dot(X.T, X) # d x d
		C = (1+self.lambdap) * np.dot(S[y, :].T, X) # a x d
		W = solve_sylvester(A,B,C) # a x d
		self.W_ = self._normalize(W, axis=1)
		self.S_ = S

	def decision_function(self, X, S):
		S_pred = np.dot(X, self.W_.T) # N x a
		# Normalize for cosine similarity
		S_pred = self._normalize(S_pred, axis = 1)

		# Normalize each attribute first
		S = self._normalize(S, axis = 0)

		# Normalize each class now # For cosine similarity
		S = self._normalize(S, axis = 1)

		return np.dot(S_pred, S.T)

	def score(self, X, S, y):
		y_pred = np.argmax(self.decision_function(X, S), axis = 1)
		return np.mean(y == y_pred)

def main():
	# for AwA dataset: Perfectly works.
	awa = scipy.io.loadmat('awa_demo_data.mat')
	train_data = awa['X_tr']
	test_data = awa['X_te']
	train_class_attributes_labels_continuous_allset = awa['S_tr']
	y_ts = awa['test_labels'] # opts.test_labels # 6180 x 1 # absolute ids.
	y_ts2 = awa['testclasses_id'] # opts.test_classes_id # 10 x 1
	test_class_attributes_labels_continuous = awa['S_te_gt']

	## Seen classes
	X_tr = awa['X_tr'] # n x d
	S_tr = awa['S_tr'] # n x a
	S_tr, y_tr = np.unique(S_tr, axis = 0, return_inverse = True) # z x a, nx1
	print('X_tr: ', X_tr.shape)
	print('Y_tr: ', y_tr.shape)
	print('S_tr: ', S_tr.shape)

	## Unseen classes
	X_ts = awa['X_te'] # n x d
	S_ts = awa['S_te_gt'] # z x a
	y_ts = awa['test_labels'][:np.newaxis] == awa['testclasses_id'][:np.newaxis].T
	y_ts = np.argmax(y_ts, axis = 1) # n x 1
	print('X_ts: ', X_ts.shape)
	print('Y_ts: ', y_ts.shape)
	print('S_ts: ', S_ts.shape)

	clf = SAE()
	clf.fit(X_tr, S_tr, y_tr)
	print('Predicting on train data')
	print(clf.score(X_tr, S_tr, y_tr))

	print('Predicting')
	print(clf.score(X_ts, S_ts, y_ts))

if __name__ == '__main__':
	main()
