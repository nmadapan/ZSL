import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.base import BaseEstimator
from platt import SigmoidTrain, SigmoidPredict
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

class SVMClassifier(BaseEstimator):
	def __init__(self, skewedness=3., n_components=85, C=100, rs = None):
		self.platt_params = []
		self.feature_map_fourier = SkewedChi2Sampler(skewedness=skewedness,	n_components=n_components, random_state = rs)
		# random_state plays a role in LinearSVC and SVC when dual = True (It is defaulted to True). 
		self.clf = Pipeline([('fp', self.feature_map_fourier),
							 ('svm', LinearSVC(C=C, random_state = rs, class_weight = 'balanced'))
							])

	def fit(self, X, y):
		self.clf.fit(X, y)

	def set_platt_params(self, X, y):
		# Use self.clf.decision_function() instead of self.clf.predict()
		# Former returns a score while latter gives the class label directly. 
		# Platt scaling transforms these scores to a probability. 
		# y_pred = self.clf.predict(X)
		y_pred = self.clf.decision_function(X)
		self.platt_params = SigmoidTrain(y_pred, y)

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		# y_pred = self.clf.predict(X)
		y_pred = self.clf.decision_function(X)
		return SigmoidPredict(y_pred, self.platt_params)

class SVMClassifierIAP(BaseEstimator):
	def __init__(self, skewedness=3., n_components=85, C=100., rs = None):
		self.feature_map_fourier = SkewedChi2Sampler(skewedness=skewedness,	n_components=n_components, random_state = rs)
		self.clf = Pipeline([("fp", self.feature_map_fourier),
			("svm", SVC(C=C, probability=True, decision_function_shape='ovr', random_state = rs))])

	def fit(self, X, y):
		self.clf.fit(X, y)

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		return self.clf.predict_proba(X)
