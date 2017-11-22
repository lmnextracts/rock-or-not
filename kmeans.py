import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mode
from numpy.linalg import det, inv, norm
from numpy.core.umath_tests import inner1d
from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# user-defined constants
GENRES = np.array([1,4], dtype = np.int)
NUM_GENRES = len(GENRES)
K_START = 3
K_END = 201
FEATURE_DIM = 15
ITERATIONS = 10

DATASET_SIZE = 1000
TRACK_COUNT_PER_GENRE = 100
CURRENT_DATASET_SIZE = NUM_GENRES * TRACK_COUNT_PER_GENRE
TRAINSET_PERCENT = 0.7
TESTSET_PERCENT = 0.3

TRAIN_SET = int(TRAINSET_PERCENT * CURRENT_DATASET_SIZE)
TEST_SET = int(TESTSET_PERCENT * CURRENT_DATASET_SIZE)

FILEPATH_DATA = 'C:\\Users\\Lakshmi\Desktop\\repo229\\rock-or-not\\data\\kmeans'
FILEPATH_PLOTS = 'C:\\Users\\Lakshmi\Desktop\\repo229\\rock-or-not\\plots\\kmeans'


FILEPATH_MEAN = os.path.join(FILEPATH_DATA,'mean_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV = os.path.join(FILEPATH_DATA, 'cov_{}.csv'.format(FEATURE_DIM))
FILEPATH_LABELS = os.path.join(FILEPATH_DATA, 'labels.csv')

FILEPATH_MEAN_TRAIN = os.path.join(FILEPATH_DATA, 'meanTrain_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV_TRAIN = os.path.join(FILEPATH_DATA, 'covTrain_{}.csv'.format(FEATURE_DIM))
FILEPATH_MEAN_TEST = os.path.join(FILEPATH_DATA, 'meanTest_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV_TEST = os.path.join(FILEPATH_DATA, 'covTest_{}.csv'.format(FEATURE_DIM))
FILEPATH_LABELS_TRAIN = os.path.join(FILEPATH_DATA, 'labelsTrain.csv')
FILEPATH_LABELS_TEST = os.path.join(FILEPATH_DATA, 'labelsTest.csv')

# Splits data into training set, dev set and test set.
# Param <genres>: Array specifying the genres to consider
def splitData(genres):
	# Read all data from file
	mean = np.loadtxt(FILEPATH_MEAN, delimiter = ',')
	cov = np.loadtxt(FILEPATH_COV, delimiter = ',')
	cov = cov.reshape(DATASET_SIZE, FEATURE_DIM, FEATURE_DIM)
	labels = np.loadtxt(FILEPATH_LABELS)

	# Extract records relevant to genres specified in the parameter <genres>
	indices = []
	for x in genres:
		idx = range(x * TRACK_COUNT_PER_GENRE, x * TRACK_COUNT_PER_GENRE + TRACK_COUNT_PER_GENRE)
		indices = np.append(indices, idx)
	indices = np.array(indices, dtype = np.int)	
	mean = mean[indices]
	cov = cov[indices]
	labels = labels[indices]
	
	# allIndex = np.array(range(0,CURRENT_DATASET_SIZE))
	# idxTrain = np.random.choice(CURRENT_DATASET_SIZE, TRAIN_SET, replace = False)
	# idxTest = np.setdiff1d(allIndex, idxTrain)

	idxTrain = []
	idxDev = []
	idxTest = []
	a = []
	b = []	
	c = []

	# Extract train and test records and save as CSV
	for x, g in enumerate(genres):
		a = range(x * TRACK_COUNT_PER_GENRE, x * TRACK_COUNT_PER_GENRE + int(TRAINSET_PERCENT * TRACK_COUNT_PER_GENRE))
		idxTrain = np.append(idxTrain,a)
		c = range(x * TRACK_COUNT_PER_GENRE + int(TRAINSET_PERCENT * TRACK_COUNT_PER_GENRE), (x+1) * TRACK_COUNT_PER_GENRE)
		idxTest = np.append(idxTest,c)

	# Extract train and test records and save as CSV
	idxTrain = np.array(idxTrain, dtype = np.int)
	idxTest = np.array(idxTest, dtype = np.int)

	# Reshape
	trainCov = cov[idxTrain].reshape(idxTrain.shape[0] * FEATURE_DIM, FEATURE_DIM)
	testCov = cov[idxTest].reshape(idxTest.shape[0] * FEATURE_DIM, FEATURE_DIM)

	# Save split data as CSV files
	np.savetxt(FILEPATH_MEAN_TRAIN,mean[idxTrain],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_MEAN_TEST,mean[idxTest],delimiter=',',newline='\n')
	
	np.savetxt(FILEPATH_COV_TRAIN, trainCov, delimiter=',', newline='\n')
	np.savetxt(FILEPATH_COV_TEST, testCov, delimiter=',', newline='\n')	

	np.savetxt(FILEPATH_LABELS_TRAIN,labels[idxTrain],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_LABELS_TEST,labels[idxTest],delimiter=',',newline='\n')	

def readData():
	# Read all training data and associated labels
	meanTrain = np.loadtxt(FILEPATH_MEAN_TRAIN, delimiter = ',')
	covTrain = np.loadtxt(FILEPATH_COV_TRAIN, delimiter = ',')
	# Reshape covariance matrix
	covTrain = covTrain.reshape(TRAIN_SET, FEATURE_DIM, FEATURE_DIM)
	labelsTrain = np.loadtxt(FILEPATH_LABELS_TRAIN)

	# Read all training data and associated labels
	meanTest = np.loadtxt(FILEPATH_MEAN_TEST, delimiter = ',')
	covTest = np.loadtxt(FILEPATH_COV_TEST, delimiter = ',')
	# Reshape covariance matrix
	covTest = covTest.reshape(TEST_SET, FEATURE_DIM, FEATURE_DIM)
	labelsTest = np.loadtxt(FILEPATH_LABELS_TEST)

	return meanTrain, covTrain, labelsTrain, meanTest, covTest, labelsTest

# param <mean>: Matrix containing mean vectors for all data points in Train Set (Dim: TRAIN_SET x FEATURE_DIM)
# param <cov>: Matrix containing covariance matrices for all data points in Train Set (Dim: TRAIN_SET x FEATURE_DIM x FEATURE_DIM)
# Method returns the mean vectors and covariance matrices for the (NUM_GENRES) cluster centroids
def kmeans(k, mean, cov):	
	# Randomly initialize cluster centroids to 4 data points
	size = len(cov)
	indices = np.random.choice(range(0,size), k)
	c_mean = mean[indices]
	c_cov = cov[indices]	

	# Initialize parameters for loop stopping
	old_dist = np.ones((size,k))
	dist = KLD(c_mean, c_cov, mean, cov)

	variant = (norm(dist - old_dist) / norm(old_dist))
	threshold = 1e-6
	counter = 1

	while (variant > threshold):
		print 'Iteration:{}\tLoop Variant: {}'.format(counter, variant)
		old_dist = dist
		
		colors = np.argmin(dist, axis=1)

		for x in xrange(0,k):
			indices = np.where(colors==x)[0]
			c_mean[x] = np.mean(mean[indices], axis = 0)
			c_cov[x] = np.mean(cov[indices], axis=0)

		dist = KLD(c_mean, c_cov, mean, cov)
		variant = (norm(dist - old_dist) / norm(old_dist))
		counter += 1

	return c_mean, c_cov

def fitKmeans(t_mean,t_cov,c_mean,c_cov):	
	dist = KLD(c_mean, c_cov, t_mean, t_cov)
	pred = np.argmin(dist, axis=1)
	return pred

# Calculate KLD using vectorization
# param <meanTrain>: TRAIN_SET X FEATURE_DIM
# param <covTrain>: TRAIN_SET X FEATURE_DIM X FEATURE_DIM
# param <meanTest>: TEST_SET X FEATURE_DIM
# param <covTrain>: TEST_SET X FEATURE_DIM X FEATURE_DIM
def KLD(meanQ, covQ, meanP, covP):
	lenP = len(meanP)
	lenQ = len(covQ)
	cP = det(covP).reshape(lenP, 1)
	cQ = det(covQ).reshape(1, lenQ)
	term1 = np.log(cQ / cP)

	iCovQ = inv(covQ)
	cP = covP
	term2 = np.einsum('qik,pkj->pijq', iCovQ, cP)
	term2 = np.einsum('piiq->pq', term2)

	diff = (meanP[:,:,None] - meanQ.T[None,:,:]).reshape(lenP, 1, FEATURE_DIM, lenQ)
	sub = np.einsum('pikq,qkj->pijq', diff, iCovQ)
	term3 = np.einsum('pikq, pkiq->pq', sub, diff)

	term4 = np.ones((lenP, lenQ)) * FEATURE_DIM

	dist = term1 + term2 + term3 - term4
	return dist

# param <data>: nd array consisting training data
# param <labels> nd array consisting of corresponding labels
# param <title> Title for the 2D plot of PCA
# This method rojects given data set onto 2D space using PCA
def visualizeData(data, labels, title):
	X = pd.DataFrame(data)	
	X_norm = (X - X.min()) / (X.max() - X.min())
	pca = sklearnPCA(n_components=2)
	transformed = pd.DataFrame(pca.fit_transform(X_norm))

	colors = ['red', 'blue', 'green', 'brown', 'yellow']
	plt.figure()

	labels = labels.astype(np.int)
	for x in xrange(0,NUM_GENRES):
		# plt.scatter(data[labels==GENRES[x]][0], data[labels==GENRES[x]][5], label='Genre{}'.format(GENRES[x]), c = colors[x])
		# Check if plot if for train/test data or fit data
		if('Clustered' in title):
			plt.scatter(X_norm[labels==x][0], X_norm[labels==x][1], label='Cluster{}'.format(x+1), c = colors[x])
		else:
			plt.scatter(X_norm[labels==GENRES[x]][0], X_norm[labels==GENRES[x]][1], label='Genre{}'.format(GENRES[x]), c = colors[x])			
	plt.legend(loc='best')
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.title(title)
	plt.savefig(os.path.join(FILEPATH_PLOTS, title + '_rockHH.png'))
	plt.show()
	return

def main():
	# Choose the genres given in GENRES
	splitData(GENRES)	
	meanTrain, covTrain, labelsTrain, meanTest, covTest, labelsTest = readData()
	visualizeData(meanTrain, labelsTrain, '2D Plot for Training Data using PCA')
	visualizeData(meanTest, labelsTest, '2D Plot for Test Data using PCA')

	# Train for kmeans using Dev-Set	
	c_mean, c_cov = kmeans(NUM_GENRES, meanTrain, covTrain)
	pred = fitKmeans(meanTest, covTest, c_mean, c_cov)
	visualizeData(meanTest, pred, '2D Plot for Clustered Data')
	return

if __name__ == '__main__':
	main()