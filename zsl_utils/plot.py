import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler

######################################################
######### Plotting confusion and roc curves ##########
######################################################

def plot_confusion(confusion, classes, wpath = ''):
	fig=plt.figure()
	plt.imshow(confusion,interpolation='nearest',origin='upper')
	plt.clim(0,1)
	plt.xticks(np.arange(0,len(classes)),[c.replace('+',' ') for c in classes],rotation='vertical',fontsize=24)
	plt.yticks(np.arange(0,len(classes)),[c.replace('+',' ') for c in classes],fontsize=24)
	plt.axis([-.5, len(classes)-.5, -.5, len(classes)-.5])
	plt.setp(plt.gca().xaxis.get_major_ticks(), pad=18)
	plt.setp(plt.gca().yaxis.get_major_ticks(), pad=12)
	fig.subplots_adjust(left=0.30)
	fig.subplots_adjust(top=0.98)
	fig.subplots_adjust(right=0.98)
	fig.subplots_adjust(bottom=0.22)
	plt.gray()
	plt.colorbar(shrink=0.79)
	if(len(wpath) == 0):
		plt.show()
	else:
		plt.savefig(wpath)
	return 

def plot_roc(P, GT, classes, wpath = ''):
	AUC=[]
	CURVE=[]
	for i,c in enumerate(classes):
		fp, tp, _ = roc_curve(GT == i,  P[:,i])
		roc_auc = auc(fp, tp)
		print ("AUC: %s %5.3f" % (c,roc_auc))
		AUC.append(roc_auc)
		CURVE.append(np.array([fp,tp]))

	print ("----------------------------------")
	print ("Mean classAUC %g" % (np.mean(AUC)*100))

	order = np.argsort(AUC)[::-1]
	styles=['-','-','-','-','-','-','-','--','--','--']
	plt.figure(figsize=(9,5))
	for i in order:
		c = classes[i]
		plt.plot(CURVE[i][0],CURVE[i][1],label='%s (AUC: %3.2f)' % (c,AUC[i]),linewidth=3,linestyle=styles[i])
	
	plt.legend(loc='lower right')
	plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
	plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
	plt.xlabel('false positive rate',fontsize=18)
	plt.ylabel('true positive rate',fontsize=18)
	if(len(wpath) == 0): plt.show()
	else: plt.savefig(wpath)
	return AUC, CURVE

def autolabel(rects, ax):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		if np.isnan(rect.get_height()):
			continue
		else:
			height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%0.3f' % height, 
			ha='center', va='bottom', rotation=90)

def plot_attAUC(P, y_true, wpath, attributes = None):
	AUC=[]
	if(attributes is None):
		attributes = map(str, range(y_true.shape[1]))

	for i in range(y_true.shape[1]):
		fp, tp, _ = roc_curve(y_true[:,i],  P[:,i])
		roc_auc = auc(fp, tp)
		AUC.append(roc_auc)
	print ("Mean attrAUC %g" % (np.nanmean(AUC)) )

	xs = np.arange(y_true.shape[1])
	width = 0.5

	# fig = plt.figure(figsize=(15,5))
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	rects = ax.bar(xs, AUC, width, align='center')
	ax.set_xticks(xs)
	ax.set_xticklabels(attributes,  rotation=90)
	ax.set_ylabel("area under ROC curve")
	autolabel(rects, ax)
	if(len(wpath) == 0): plt.show()
	else: plt.savefig(wpath)
	return AUC