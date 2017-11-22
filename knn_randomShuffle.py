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
K_START = 3
K_END = 201
FEATURE_DIM = 15
ITERATIONS = 10

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
FILEPATH_MEAN_DEV = os.path.join(FILEPATH_DATA, 'meanDev_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV_DEV = os.path.join(FILEPATH_DATA, 'covDev_{}.csv'.format(FEATURE_DIM))
FILEPATH_MEAN_TEST = os.path.join(FILEPATH_DATA, 'meanTest_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV_TEST = os.path.join(FILEPATH_DATA, 'covTest_{}.csv'.format(FEATURE_DIM))
FILEPATH_LABELS_TRAIN = os.path.join(FILEPATH_DATA, 'labelsTrain.csv')
FILEPATH_LABELS_TEST = os.path.join(FILEPATH_DATA, 'labelsTest.csv')
FILEPATH_LABELS_DEV = os.path.join(FILEPATH_DATA, 'labelsDev.csv')


def printDecorated(message):
	print '\n**********************************************'
	print message
	print '\n**********************************************'
	return

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
		b = range(x * TRACK_COUNT_PER_GENRE + int(TRAINSET_PERCENT * TRACK_COUNT_PER_GENRE), x * TRACK_COUNT_PER_GENRE + int((TRAINSET_PERCENT + DEVSET_PERCENT) * TRACK_COUNT_PER_GENRE))
		idxDev = np.append(idxDev,b)
		c = range(x * TRACK_COUNT_PER_GENRE + int( (TRAINSET_PERCENT + DEVSET_PERCENT) * TRACK_COUNT_PER_GENRE), (x+1) * TRACK_COUNT_PER_GENRE)
		idxTest = np.append(idxTest,c)

	# Extract train and test records and save as CSV
	idxTrain = np.array(idxTrain, dtype = np.int)
	idxDev = np.array(idxDev, dtype = np.int)
	idxTest = np.array(idxTest, dtype = np.int)

	# Reshape
	trainCov = cov[idxTrain].reshape(idxTrain.shape[0] * FEATURE_DIM, FEATURE_DIM)
	devCov = cov[idxDev].reshape(idxDev.shape[0] * FEATURE_DIM, FEATURE_DIM)
	testCov = cov[idxTest].reshape(idxTest.shape[0] * FEATURE_DIM, FEATURE_DIM)

	# Save split data as CSV files
	np.savetxt(FILEPATH_MEAN_TRAIN,mean[idxTrain],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_MEAN_DEV,mean[idxDev],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_MEAN_TEST,mean[idxTest],delimiter=',',newline='\n')
	
	np.savetxt(FILEPATH_COV_TRAIN, trainCov, delimiter=',', newline='\n')
	np.savetxt(FILEPATH_COV_DEV, devCov, delimiter=',', newline='\n')
	np.savetxt(FILEPATH_COV_TEST, testCov, delimiter=',', newline='\n')	

	np.savetxt(FILEPATH_LABELS_TRAIN,labels[idxTrain],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_LABELS_DEV,labels[idxDev],delimiter=',',newline='\n')
	np.savetxt(FILEPATH_LABELS_TEST,labels[idxTest],delimiter=',',newline='\n')	

# Calculate KLD using vectorization
# param <meanTrain>: TRAIN_SET X FEATURE_DIM
# param <covTrain>: TRAIN_SET X FEATURE_DIM X FEATURE_DIM
# param <meanTest>: TEST_SET X FEATURE_DIM
# param <covTrain>: TEST_SET X FEATURE_DIM X FEATURE_DIM
def KLD(meanTrain, covTrain, meanTest, covTest):
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
	return dist

# Calculate KL Divergence between two songs given their mean vectors and covariance matrices
def KLDivergence(muP, muQ, covP, covQ):
	iCovQ = inv(covQ)
	diff = muP - muQ
	# len(covP) gives the number of dimensions, d
	return np.log(det(covQ)/det(covP)) + np.trace(np.matmul(iCovQ, covP)) + np.matmul(diff.T, np.matmul(iCovQ, diff)) - len(covP)

def readData():
	# Read all training data and associated labels
	meanTrain = np.loadtxt(FILEPATH_MEAN_TRAIN, delimiter = ',')
	covTrain = np.loadtxt(FILEPATH_COV_TRAIN, delimiter = ',')
	# Reshape covariance matrix
	covTrain = covTrain.reshape(TRAIN_SET, FEATURE_DIM, FEATURE_DIM)
	labelsTrain = np.loadtxt(FILEPATH_LABELS_TRAIN)

	# Read all training data and associated labels
	meanDev = np.loadtxt(FILEPATH_MEAN_DEV, delimiter = ',')
	covDev = np.loadtxt(FILEPATH_COV_DEV, delimiter = ',')
	# Reshape covariance matrix
	covDev = covDev.reshape(DEV_SET, FEATURE_DIM, FEATURE_DIM)
	labelsDev = np.loadtxt(FILEPATH_LABELS_DEV)

	# Read all training data and associated labels
	meanTest = np.loadtxt(FILEPATH_MEAN_TEST, delimiter = ',')
	covTest = np.loadtxt(FILEPATH_COV_TEST, delimiter = ',')
	# Reshape covariance matrix
	covTest = covDev.reshape(TEST_SET, FEATURE_DIM, FEATURE_DIM)
	labelsTest = np.loadtxt(FILEPATH_LABELS_TEST)

	return meanTrain, covTrain, labelsTrain, meanDev, covDev, labelsDev, meanTest, covTest, labelsTest

def knnKLD(k, meanTrain, covTrain, labelsTrain, meanTest, covTest, labelsTest):
	testSet = len(meanTest)
	trainSet = len(meanTrain)

	dist = np.zeros((testSet, trainSet))
	pred = np.zeros(testSet)
	labelsDev = labelsTest[0:testSet].astype(int)
	accuracy = 0

	for x in xrange(0,testSet):
		print 'Classifying record {}'.format(x)

		#  Find distance of data point with every song in the training set
		for y in xrange(0,trainSet):
			dist[x,y] = KLDivergence(meanTest[x], meanTrain[y], covTest[x], covTrain[y])		


	# # Find the k nearest songs
	# knearest = np.argsort(dist[x])[0:k]
	# knearest = labelsTrain[knearest].astype(int)

	# # Classify song with the maximum label count
	# counts = np.bincount(knearest)
	# pred[x] = np.argmax(counts)
	# pred = pred.astype(int)
	# # Calculate accuracy
	# accuracy = np.where(labelsDev == pred)[0].shape[0] * 1. / testSet

	print 'Accuracy: {}, \tFor K={}'.format(accuracy, k)
	
	return pred

def knn(k, meanTrain, covTrain, labelsTrain, meanTest, covTest):
	dist = KLD(meanTrain, covTrain, meanTest, covTest)

	neighbours = np.argsort(dist, axis=1)[:,range(0,k)]
	# neighbours = np.array(map(lambda x: labelsTrain[x], neighbours))
	neighbours = labelsTrain[neighbours]
	pred = np.array(mode(neighbours, axis=1)[0]).reshape(len(meanTest),)	

	return pred

def main():

	# # Training with Dev for Vectorized KNN
	# accuracy = np.zeros(K_END - K_START)

	# Choose the following genres: Classical, HipHop, Jazz and Rock
	# splitData(GENRES)		
	# meanTrain, covTrain, labelsTrain, meanDev, covDev, labelsDev, meanTest, covTest, labelsTest = readData()

	# # Find optimal K by using the Dev-Set
	# for k in xrange(K_START, K_END):				
	# 	pred = knn(k, meanTrain, covTrain, labelsTrain, meanDev, covDev)
	# 	accuracy[k-3] = np.where(pred == labelsDev)[0].shape[0] * 1. / len(meanDev)
	# 	print 'k={}\tAccuracy: {}'.format(k,accuracy[k-3])

	# optimalK = int(np.argmax(accuracy) + 3)
	# print '\n k-optimal: {} with accuracy = {}'.format(optimalK, accuracy[optimalK])

	# # # Plot k-Value vs Accuracy
	# plt.figure()
	# plt.plot(range(K_START,K_END), accuracy)	
	# plt.xlabel('K Value')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy vs K Value for Dev-Set')
	# plt.savefig(os.path.join(FILEPATH_PLOTS, 'kValueVsAccuracyFinal.png'))

	# Choose the following genres: Classical, HipHop, Jazz and Rock
	splitData(GENRES)		
	accuracy = np.zeros(K_END - K_START)
	optimalK = np.zeros(ITERATIONS)

	meanTrain, covTrain, labelsTrain, meanDev, covDev, labelsDev, meanTest, covTest, labelsTest = readData()

	for x in xrange(0, ITERATIONS):
		printDecorated('Iteration {}'.format(x+1))

		# Find optimal K by using the Dev-Set
		for k in xrange(K_START, K_END):				
			pred = knn(k, meanTrain, covTrain, labelsTrain, meanDev, covDev)
			accuracy[k-3] = np.where(pred == labelsDev)[0].shape[0] * 1. / len(meanDev)
			# pred = knn(k, meanTrain, covTrain, labelsTrain, meanTest, covTest)
			# accuracy[k-3] = np.where(pred == labelsTest)[0].shape[0] * 1. / len(meanTest)
			print 'k={}\tAccuracy: {}'.format(k,accuracy[k-3])

		optimalK[x] = int(np.argmax(accuracy) + 3)
		print '\n k-optimal: {} with accuracy = {}'.format(optimalK[x], accuracy[int(optimalK[x]-3)])

		# # Plot k-Value vs Accuracy
		plt.figure()
		plt.plot(range(K_START,K_END), accuracy)	
		plt.xlabel('K Value')
		plt.ylabel('Accuracy')
		plt.title('Accuracy vs K Value for Dev-Set')
		plt.savefig(os.path.join(FILEPATH_PLOTS, 'kVsAccuracy{}.png'.format(x+1)))		

	kValueForTest = int(np.mean(optimalK))
	
	# Comment out the part above and run with kValueForTest replaced by optimal K obtained from above.
	# Predict labels for test set using k-optimal 
	pred = knn(optimalK, meanTrain, covTrain, labelsTrain, meanTest, covTest)
	# pred = knnKLD(kValueForTest, meanTrain, covTrain, labelsTrain, meanTest, covTest, labelsTest)
	accuracyTest = np.where(pred == labelsTest)[0].shape[0] * 1. / len(meanTest)
	print 'Accuracy: {}'.format(accuracyTest)
	print 'Confusion Matrix'
	print confusion_matrix(labelsTest, pred)

if __name__ == '__main__':
	main()
