import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

from .utils import CustomScaler
from .platt import SigmoidTrain, SigmoidPredict

import warnings
warnings.filterwarnings('ignore')

class SVMClassifier(BaseEstimator):
	def __init__(self, skewedness=3., n_components=85, C=100, clamp =3., rs = None):
		self.platt_params = []
		self.feature_map_fourier = SkewedChi2Sampler(skewedness=skewedness,	n_components=n_components, random_state = rs)
		self.c_scaler = CustomScaler(clamp = clamp-0.1)
		# random_state plays a role in LinearSVC and SVC when dual = True (It is defaulted to True). 
		self.clf_ = Pipeline([('cs', self.c_scaler),
							 ('fp', self.feature_map_fourier),
							 ('svm', LinearSVC(C=C, random_state = rs, class_weight = 'balanced'))
							])
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
	def __init__(self, skewedness=3., n_components=85, C=100., clamp =3., rs = None):
		self.c_scaler = CustomScaler(clamp = clamp-0.1)
		self.feature_map_fourier = SkewedChi2Sampler(skewedness=skewedness,	n_components=n_components, random_state = rs)
		self.clf_ = Pipeline([('cs', self.c_scaler),
			("fp", self.feature_map_fourier),
			("svm", SVC(C=C, probability=True, decision_function_shape='ovr', random_state = rs))])

	def fit(self, X, y):
		self.clf_.fit(X, y)

	def predict(self, X):
		return self.clf_.predict(X)

	def predict_proba(self, X):
		return self.clf_.predict_proba(X)
