# -*- coding: utf-8 -*-
"""
@author: Dr. Fayyaz Minhas
@author-email: afsar at pieas dot edu dot pk
Implementation of embarrasingly simple zero shot learning
"""
from __future__ import print_function
from numpy.random import randn #importing randn
import numpy as np #importing numpy
from plotit import plotit

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
	
def accuracy(ytarget,ypredicted):
	return np.mean(ytarget == ypredicted)

def getExamples(n=100,d=2):
	"""
	Generates n d-dimensional normally distributed examples of each class        
	The mean of the positive class is [1] and for the negative class it is [-1]
	DO NOT CHANGE THIS FUNCTION
	"""
	Xp = randn(n,d)#+1   #generate n examples of the positie class
	Xp=Xp
	Xn = randn(int(n/2),d)#-1   #generate n examples of the negative class
	Xn=Xn-5
	Xn2 = randn(int(n/2),d)#-1   #generate n examples of the negative class
	Xn2=Xn2+5
	Xn = np.vstack((Xn,Xn2))
	X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
	Y = np.array([0]*n+[1]*int(n/2)+[2]*int(n/2)) #Associate Labels
	return (X,Y) 


def poly(X1,X2,**kwargs):
	if 'degree' not in kwargs:
		d = 1
	else:
		d = kwargs['degree']
		
	return (np.dot(X1,X2.T)+1)**d

class ESZSL(ClassifierMixin, BaseEstimator):
	def __init__(self, lambdap = 0.1, sigmap = 0.1, kernel = poly, **kwargs):
		"""
		lambdap: Regularization parameter for kernel/feature space
		sigmap: Regularization parameter for Attribute Space
		kernel: kernel function (default is poly)
		kwargs: optional, any kernel arguments
		"""
		self.sigmap = sigmap
		self.lambdap = lambdap
		self.kwargs = kwargs
		self.kernel = kernel

		# self.A_ = None
		
	def fit(self, X, Y, S=None):
		if S is None:
			S = np.eye(Y.shape[1])

		if self.kernel=='precomputed':
			K = X
		else:            
			self.X = X
			K = self.kernel(X,X,**self.kwargs) 
		KK = np.dot(K.T,K)
		KK = np.linalg.inv(KK+self.lambdap*(np.eye(K.shape[0])))
		KYS = np.dot(np.dot(K,Y),S)    
		SS = np.dot(S.T,S)
		SS = np.linalg.inv(SS+self.sigmap*np.eye(SS.shape[0]))
		self.A_ = np.dot(np.dot(KK,KYS),SS)
		
	def decision_function(self,X,S = None):
		if S is None:
			S = np.eye(self.A_.shape[1])[0,:]

		if self.kernel=='precomputed':
			K = X
		else:
			K = self.kernel(X,self.X,**self.kwargs)
		Z = np.dot(np.dot(S,self.A_.T),K.T).T
		return Z
	
	def predict(self,X,S=None):
		return np.argmax(self.decision_function(X,S),axis=1)

	def score(self, X, y_true, S=None):
		y_pred = self.predict(X, S)
		return np.mean(y_true == y_pred)
		
if __name__ == '__main__':
#%% Data Generation for simple classification 

	n = 500 #number of examples of each class
	d = 2 #number of dimensions
	Xtr,Ytr = getExamples(n=n,d=d) #Generate Training Examples    
	print("Number of positive examples in training: ", np.sum(Ytr==1))
	print("Number of negative examples in training: ", np.sum(Ytr==-1))
	print("Dimensions of the data: ", Xtr.shape[1])   
	Xtt,Ytt = getExamples(n=100,d=d) #Generate Testing Examples 
	z  = len(set(Ytt))
	#%% Setting up classlabel matrix Y and attribute matrix S for binary classification
	Y = -1*np.ones((Xtr.shape[0],z))
	for i in range(len(Y)):
		Y[i,Ytr[i]]=1
	S = np.eye(z,z)
	
	
		 
	#%% Training and evaluation, plotting
	#from sklearn.metrics.pairwise import polynomial_kernel as poly #use builtin-kernel functions from sk-learn
	#from sklearn.metrics.pairwise import rbf_kernel as rbf
	clf = ESZSL(sigmap = 0.1, lambdap = 0.05, kernel = poly, degree = 1)
	clf.fit(Xtr,Y,S)
	Z = clf.decision_function(Xtr,S)[:,1]
	print("Train accuracy",accuracy(Ytr,clf.predict(Xtr,S)))
	Z = clf.decision_function(Xtt,S)[:,1]
	print("Train accuracy",accuracy(Ytt,clf.predict(Xtt,S)))
	plotit(Xtr,Ytr,clf=clf.predict,S=S,colors='random')    
	
#%% Training and evaluation for precomputed matrix
	K = (np.dot(Xtr,Xtr.T)+1)**2
	clf = ESZSL(sigmap = 0.1, lambdap = 0.1, kernel = 'precomputed')
	clf.fit(K,Y,S)
	Ktt = (np.dot(Xtt,Xtr.T)+1)**2
	print("Train accuracy",accuracy(Ytr,clf.predict(K,S)))
	print("Test accuracy",accuracy(Ytt,clf.predict(Ktt,S)))
	

