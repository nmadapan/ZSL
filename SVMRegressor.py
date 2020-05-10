import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.base import BaseEstimator
from platt import SigmoidTrain, SigmoidPredict

from utils import CustomScaler

class SVMRegressor(BaseEstimator):
	def __init__(self, skewedness=3., n_components=85, C=100, clamp =3., rs=None):
		self.platt_params = []
		self.c_scaler = CustomScaler(clamp = clamp-0.1)
		self.feature_map_fourier = SkewedChi2Sampler(skewedness=skewedness,	n_components=n_components, random_state=rs)
		self.clf = Pipeline([('cs', self.c_scaler),
							 ("fp", self.feature_map_fourier),
							 ("svm", SVR(C=C))
							])

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
	def __init__(self, skewedness=3., n_components=85, C=100., clamp =3., rs=None):
		self.feature_map_fourier = SkewedChi2Sampler(skewedness=skewedness,	n_components=n_components, random_state=rs)
		self.c_scaler = CustomScaler(clamp = clamp-0.1)
		self.clf = Pipeline([('cs', self.c_scaler),
			("fp", self.feature_map_fourier),
			("svm", SVR(C=C))]) ## TODO: Check it. There is a bug in the original code. 

	def fit(self, X, y):
		self.clf.fit(X, y)

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		return self.clf.predict_proba(X)
