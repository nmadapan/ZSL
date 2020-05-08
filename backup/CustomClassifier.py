from numpy.random import randn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.estimator_checks import check_estimator

class CustomClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, C = 10.0):
		self.C = C
		self.clf_ = LinearSVC(C = self.C)

	def fit(self, X, y): #2
		print('Fit: ', self.C)
		self.clf_.fit(X, y)
		return self

	def decision_function(self, X):
		print('Decision function: ', self.C)
		return self.clf_.decision_function(X)

	def predict(self, X):
		print('Predict: ', self.C)
		return self.clf_.predict(X)

	def score(self, X, y): #3
		print('Score: ', self.C)
		## Fit should be called before calling score()
		# X need not be training data. it can be anyother data. 
		return self.clf_.score(X, y)

	def set_params(self, **parameters): #1
		print(parameters)
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		self.clf_.set_params(**parameters)
		return self

if __name__ == '__main__':
	iris = datasets.load_iris()

	# Take the first two features. We could avoid this by using a two-dim dataset
	X = iris.data[:, :2]
	y = iris.target
	temp = np.logical_or(y == 0, y == 1)
	y = y[temp]
	X = X[temp, :]

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state = 42)
	model = CustomClassifier()

	clf = GridSearchCV(model, {'C': [1e-1, 1e1, 1e2]}, cv = 5, n_jobs=None)
	clf.fit(x_train, y_train)
	print(clf.best_estimator_)

	print(check_estimator(CustomClassifier))