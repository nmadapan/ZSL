import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class Preprocess(BaseEstimator, TransformerMixin):
	def __init__(self, clamp = 3.0):
		self.clamp = clamp

	def fit(self, X):
		self.mean_ = np.mean(X, axis=0)
		self.std_ = np.std(X, axis=0)
		return self

	def transform(self, X):
		X = np.copy(X) - self.mean_
		X /= self.std_
		X[X > self.clamp] = self.clamp
		X[X < -1 * self.clamp] = -1 * self.clamp
		return X

	def inverse_transform(self, X):
		X = np.copy(X) * self.std_
		X += self.mean_
		return X

class Preprocess2(StandardScaler):
	def __init__(self, clamp = 3.0):
		super().__init__()
		self.clamp = clamp

	def transform(self, X):
		X = np.copy(X) - self.mean_
		X /= self.scale_
		X[X > self.clamp] = self.clamp
		X[X < -1 * self.clamp] = -1 * self.clamp
		return X

if __name__ == '__main__':
	clf = Preprocess()
	clf2 = Preprocess2()
	print(Preprocess2.__mro__)
	X = np.random.randint(0, 10, (3, 4))
	A = np.random.randint(0, 10, (3, 4))
	
	clf.fit(X)
	print(X)
	print(clf.mean_, clf.std_)
	print(A)
	print(clf.transform(A))

	print('###############')
	clf2.fit(X)
	print(X)
	print(clf2.mean_, clf2.scale_)
	print(A)
	print(clf2.transform(A))