import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.base import BaseEstimator

from .platt import SigmoidTrain, SigmoidPredict

class SVMRegressor(BaseEstimator):
	def __init__(self, C=100, rs=None):
		self.platt_params = []
		self.clf = SVR(C=C)

	def fit(self, X, y):
		self.clf.fit(X, y)

	def set_platt_params(self, X, y):
		y_pred = self.clf.predict(X) # For SVR, predict() gives a score. 
		self.platt_params = SigmoidTrain(y_pred, y)

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		y_pred = self.clf.predict(X)
		return SigmoidPredict(y_pred, self.platt_params)

class SVMRegressorIAP(BaseEstimator):
	def __init__(self, C=100., rs=None):
		self.clf = SVR(C=C) ## TODO: Check it. There is a bug in the original code. 

	def fit(self, X, y):
		self.clf.fit(X, y)

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		return self.clf.predict_proba(X)
