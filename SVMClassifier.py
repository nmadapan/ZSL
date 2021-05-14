import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

from .platt import SigmoidTrain, SigmoidPredict

import warnings
warnings.filterwarnings('ignore')

class SVMClassifier(BaseEstimator):
	def __init__(self, C=100, rs = None):
		self.platt_params = []
		self.clf_ = LinearSVC(C=C, random_state = rs, class_weight = 'balanced')
		self.dummy_ = False

	def fit(self, X, y):
		if(len(np.unique(y)) == 1): 
			self.clf_ = DummyClassifier(strategy="most_frequent").fit(X, y)
			self.dummy_ = True
		else: 
			self.clf_.fit(X, y)

	def set_platt_params(self, X, y):
		# Use self.clf_.decision_function() instead of self.clf_.predict()
		# Former returns a score while latter gives the class label directly. 
		# Platt scaling transforms these scores to a probability. 
		if(self.dummy_):
			self.platt_params = None
		else:	
			y_pred = self.clf_.decision_function(X)
			self.platt_params = SigmoidTrain(y_pred, y)

	def predict(self, X):
		return self.clf_.predict(X)

	def predict_proba(self, X):
		if(self.dummy_):
			return self.clf_.predict_proba(X).flatten()
		else:
			y_pred = self.clf_.decision_function(X)
			return SigmoidPredict(y_pred, self.platt_params)

class SVMClassifierIAP(BaseEstimator):
	def __init__(self, C=100., rs = None):
		self.clf_ = SVC(C=C, probability=True, decision_function_shape='ovr', random_state = rs)

	def fit(self, X, y):
		self.clf_.fit(X, y)

	def predict(self, X):
		return self.clf_.predict(X)

	def predict_proba(self, X):
		return self.clf_.predict_proba(X)
