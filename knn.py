import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mode
from numpy.linalg import det, inv
from numpy.core.umath_tests import inner1d
from sklearn.metrics import confusion_matrix

#debug
SEP = '*****************************************\n\n***************************************'

# user-defined constants
DIM = 20
GENRES = np.array([1,4,5,9], dtype = np.int)
KVALUES = 198
K_START = 3
K_END = 200
FEATURE_DIM = 15

DATASET_SIZE = 1000
TRACK_COUNT_PER_GENRE = 100
CURRENT_DATASET_SIZE = len(GENRES) * TRACK_COUNT_PER_GENRE
TRAINSET_PERCENT = 0.7
DEVSET_PERCENT = 0.15
TESTSET_PERCENT = 0.15

DEV_SET = int(DEVSET_PERCENT * CURRENT_DATASET_SIZE)
TRAIN_SET = int(TRAINSET_PERCENT * CURRENT_DATASET_SIZE)
TEST_SET = int(TESTSET_PERCENT * CURRENT_DATASET_SIZE)


FILEPATH_DATA = 'C:\\Users\\Lakshmi\Desktop\\repo229\\rock-or-not\\data'
FILEPATH_PLOTS = 'C:\\Users\\Lakshmi\Desktop\\repo229\\rock-or-not\\plots'


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
	
	idxTrain = []
	idxTest = []
	a = []
	b = []	

	# Extract train and test records and save as CSV
	for x, g in enumerate(genres):
		a = range(x * TRACK_COUNT_PER_GENRE, x * TRACK_COUNT_PER_GENRE + int(TRAINSET_PERCENT * TRACK_COUNT_PER_GENRE))
		idxTrain = np.append(idxTrain,a)
		b = range(x * TRACK_COUNT_PER_GENRE + int(TRAINSET_PERCENT * TRACK_COUNT_PER_GENRE), (x+1) * TRACK_COUNT_PER_GENRE)
		idxTest = np.append(idxTest,b)

	idxTrain = np.array(idxTrain, dtype = np.int)
	idxTest = np.array(idxTest, dtype = np.int)

	trainCov = cov[idxTrain].reshape(idxTrain.shape[0] * FEATURE_DIM, FEATURE_DIM)
	testCov = cov[idxTest].reshape(idxTest.shape[0] * FEATURE_DIM, FEATURE_DIM)

	np.savetxt(FILEPATH_MEAN_TRAIN,mean[idxTrain],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_MEAN_TEST,mean[idxTest],delimiter=',',newline='\n')
	
	np.savetxt(FILEPATH_COV_TRAIN, trainCov, delimiter=',', newline='\n')
	np.savetxt(FILEPATH_COV_TEST, testCov, delimiter=',', newline='\n')	

	np.savetxt(FILEPATH_LABELS_TRAIN,labels[idxTrain],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_LABELS_TEST,labels[idxTest],delimiter=',',newline='\n')
	

# Calculate KL Divergence between two songs given their mean vectors and covariance matrices
def KLDivergence(muP, muQ, covP, covQ):
	iCovQ = inv(covQ)
	diff = muP - muQ

	# debug
	# print np.log(det(covQ)/det(covP)) 
	# print np.trace(np.matmul(iCovQ, covP)) 
	# print np.matmul(diff.T, np.matmul(iCovQ, diff)) 
	# print len(covP)

	# len(covP) gives the number of dimensions, d
	return np.log(det(covQ)/det(covP)) + np.trace(np.matmul(iCovQ, covP)) + np.matmul(diff.T, np.matmul(iCovQ, diff)) - len(covP)

# Calculate KLD using vectorization
# param <meanTrain>: TRAIN_SET X FEATURE_DIM
# param <covTrain>: TRAIN_SET X FEATURE_DIM X FEATURE_DIM
# param <meanTest>: TEST_SET X FEATURE_DIM
# param <covTrain>: TEST_SET X FEATURE_DIM X FEATURE_DIM
def KLD(meanTrain, covTrain, meanTest, covTest):
	# Term1: log(det(covQ)/det(covP))
	# Term4: TEST_SET X TRAIN_SET with values FEATURE_DIM

	testSet = len(meanTest)
	trainSet = TRAIN_SET

	covP = det(covTest).reshape(testSet, 1)
	covQ = det(covTrain).reshape(1, trainSet)
	term1 = np.log(covQ / covP)

	iCovQ = inv(covTrain)
	covP = covTest
	term2 = np.einsum('qik,pkj->pijq', iCovQ, covP)
	term2 = np.einsum('piiq->pq', term2)

	diff = (meanTest[:,:,None] - meanTrain.T[None,:,:]).reshape(testSet, 1, FEATURE_DIM, trainSet)
	sub = np.einsum('pikq,qkj->pijq', diff, iCovQ)
	term3 = np.einsum('pikq, pkiq->pq', sub, diff)

	term4 = np.ones((testSet, trainSet)) * FEATURE_DIM

	dist = term1 + term2 + term3 - term4
	
	# # debug
	# print term1[0,0]
	# print term2[0,0]
	# print term3[0,0]
	# print term4[0,0]
	# print dist.shape

	return dist

def readData():
	# Read all training data and associated labels
	meanTrain = np.loadtxt(FILEPATH_MEAN_TRAIN, delimiter = ',')
	covTrain = np.loadtxt(FILEPATH_COV_TRAIN, delimiter = ',')
	# Reshape covariance matrix
	covTrain = covTrain.reshape(TRAIN_SET, FEATURE_DIM, FEATURE_DIM)
	labelsTrain = np.loadtxt(FILEPATH_LABELS_TRAIN)


	# Read all test data
	meanTest = np.loadtxt(FILEPATH_MEAN_TEST, delimiter = ',')
	meanDev = meanTest[0:DEV_SET]
	meanTest = meanTest[DEV_SET:(DEV_SET+TEST_SET)]

	covTest = np.loadtxt(FILEPATH_COV_TEST, delimiter = ',')
	covTest = covTest.reshape((DEV_SET+TEST_SET), FEATURE_DIM, FEATURE_DIM)

	covDev = covTest[0:DEV_SET].copy()
	covTest = covTest[DEV_SET:(DEV_SET+TEST_SET)]
	covDev = covDev.reshape(DEV_SET, FEATURE_DIM, FEATURE_DIM)	
	covTest = covTest.reshape(TEST_SET, FEATURE_DIM, FEATURE_DIM)	

	labelsTest = np.loadtxt(FILEPATH_LABELS_TEST)
	labelsDev = labelsTest[0:DEV_SET]
	labelsTest = labelsTest[DEV_SET:(DEV_SET+TEST_SET)]

	return meanTrain, covTrain, labelsTrain, meanDev, covDev, labelsDev, meanTest, covTest, labelsTest

def knnKLD(meanTrain, covTrain, labelsTrain, meanTest, covTest, labelsTest):
	testSet = len(meanTest)
	trainSet = len(meanTrain)

	dist = np.zeros((testSet, trainSet))
	pred = np.zeros(testSet)
	labelsDev = labelsTest[0:testSet].astype(int)

	accuracy = np.zeros(18)
	
	for K_VALUE in xrange(3,21):
		print 'Classifying for K={}'.format(K_VALUE)
		for x in xrange(0,testSet):
			#  Find distance of data point with every song in the training set
			for y in xrange(0,trainSet):
				dist[x,y] = KLDivergence(meanTest[x], meanTrain[y], covTest[x], covTrain[y])		

			# Find the k nearest songs
			knearest = np.argsort(dist[x])[0:K_VALUE]
			knearest = labelsTrain[knearest].astype(int)
			# Classify song with the maximum label count
			counts = np.bincount(knearest)
			pred[x] = np.argmax(counts)
			# print 'Classifying record {} as Genre[{}]'.format(x+1, pred[x])

			pred = pred.astype(int)
			# Calculate accuracy
			accuracy[K_VALUE-3] = np.where(labelsDev == pred)[0].shape[0] * 1. / testSet
		print 'Accuracy for K={}: {}'.format(K_VALUE, accuracy[K_VALUE-3])		

	print 'Best Accuracy obtained for K={}'.format(np.argmax(accuracy)+3)
	np.savetxt(os.path.join(FILEPATH_DATA,'accuracy3to20.csv'), accuracy)
	plt.plot(range(3,21),accuracy)
	plt.show()

# This implementation uses vectorized KLD method to determine the KL distance
def knn(k, meanTrain, covTrain, labelsTrain, meanTest, covTest):
	dist = KLD(meanTrain, covTrain, meanTest, covTest)

	neighbours = np.argsort(dist, axis=1)[:,range(0,k)]
	# neighbours = np.array(map(lambda x: labelsTrain[x], neighbours))
	neighbours = labelsTrain[neighbours]
	pred = np.array(mode(neighbours, axis=1)[0]).reshape(len(meanTest),)	
	
	return pred

def main():
	# Choose the following genres: Classical, HipHop, Jazz and Rock
	splitData(GENRES)

	# Fit KNN
	meanTrain, covTrain, labelsTrain, meanDev, covDev, labelsDev, meanTest, covTest, labelsTest = readData()
	accuracy = np.zeros(KVALUES)

	# Find optimal K by using the Dev-Set
	# for k in xrange(K_START,K_END+1):				
	# 	pred = knn(k, meanTrain, covTrain, labelsTrain, meanDev, covDev)
	# 	accuracy[k-3] = np.where(pred == labelsDev)[0].shape[0] * 1. / len(meanDev)
	# 	print 'k={}\tAccuracy: {}'.format(k,accuracy[k-3])

	# # Plot k-Value vs Accuracy
	# plt.figure()
	# plt.plot(range(K_START,K_END+1), accuracy)	
	# plt.xlabel('K Value')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy vs K Value for Dev-Set')
	# plt.savefig(os.path.join(FILEPATH_PLOTS, 'kVsAccuracy.png'))
	# plt.show()

	# kOptimal = np.argsort(accuracy)[-1] + 3
	# print '\n k-optimal: {} with accuracy = {}'.format(kOptimal, accuracy[kOptimal-3])

	# Predict labels for test set using k-optimal 
	pred = knn(kOptimal, meanTrain, covTrain, labelsTrain, meanTest, covTest)
	accuracyTest = np.where(pred == labelsTest)[0].shape[0] * 1. / len(meanTest)
	print pred
	print labelsTest
	print 'Accuracy: {}'.format(accuracyTest)
	print 'Confusion Matrix'
	print confusion_matrix(labelsTest, pred)


if __name__ == '__main__':
	main()
