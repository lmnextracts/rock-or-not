import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mode

def convertToFeatureMatrix():
	meanData = np.loadtxt('mean_15.csv', delimiter=',')
	covData = np.loadtxt('cov_15.csv', delimiter=',')

	print meanData.shape
	print covData.shape

	covData.shape = (400,20,20)

	# upperTriangular = np.triu_indices(20)
	# cov[:,] = covData[:,upperTriangular]

	print cov.shape

	# data = np.zeros((400,230))	
	# data[:,0:19] = meanData
	# data[:,20:] = covData[:]
	return
def main():
	return