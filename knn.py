import numpy as np
import os
from numpy.linalg import det, inv

# user-defined constants
DIM = 20
K_VALUE = 5
FEATURE_DIM = 15

DATASET_SIZE = 1000
TRACK_COUNT_PER_GENRE = 100
TRAINSET_PERCENT = 0.7
DEVSET_PERCENT = 0.15
TESTSET_PERCENT = 0.15

DEV_SET = DEVSET_PERCENT * DATASET_SIZE
TEST_SET = TESTSET_PERCENT * DATASET_SIZE
TRAIN_SET = TRAINSET_PERCENT * DATASET_SIZE

FILEPATH_DATA = 'C:\\Users\\Lakshmi\Desktop\\repo229\\rock-or-not\\data'

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
	# len(covP) gives the number of dimensions, d
	return np.log(det(covQ)/det(covP)) + np.trace(np.matmul(iCovQ, covP)) + np.matmul(diff.T, np.matmul(iCovQ, diff)) - len(covP)

def readData():
	# Read all training data and associated labels
	meanTrain = np.loadtxt('meanTrain_{}.csv'.format(DIM), delimiter = ',')
	covTrain = np.loadtxt('covTrain_{}.csv'.format(DIM), delimiter = ',')
	labels = np.loadtxt('labels.csv')
	# Reshape covariance matrix
	covTrain = covTrain.reshape(700, DIM, DIM)

	# Read all test data
	meanTest = np.loadtxt('meanTest_{}.csv'.format(DIM), delimiter = ',')
	covTest = np.loadtxt('covTest_{}.csv'.format(DIM), delimiter = ',')
	labelsTest = np.loadtxt('labelsTest.csv')
	# Reshape covariance matrix
	covTest = covTest.reshape(300, DIM, DIM)

	return meanTrain, covTrain, labels, meanTest, covTest, labelsTest

def knnKLD(meanTrain, covTrain, labels, meanTest, covTest, labelsTest):
	dist = np.zeros((DEV_SET, TRAIN_SET))
	pred = np.zeros(DEV_SET)
	labelsDev = labelsTest[0:DEV_SET].astype(int)

	accuracy = np.zeros(497)
	
	for K_VALUE in xrange(3,500):
		print 'Classifying for K={}'.format(K_VALUE)
		for x in xrange(0,DEV_SET):
			#  Find distance of data point with every song in the training set
			for y in xrange(0,TRAIN_SET):
				dist[x,y] = KLDivergence(meanTest[x], meanTrain[y], covTest[x], covTrain[y])		

			# Find the k nearest songs
			knearest = np.argsort(dist[x])[0:K_VALUE]
			knearest = labels[knearest].astype(int)
			# Classify song with the maximum label count
			counts = np.bincount(knearest)
			pred[x] = np.argmax(counts)
			# print 'Classifying record {} as Genre[{}]'.format(x+1, pred[x])

			pred = pred.astype(int)
			# Calculate accuracy
			accuracy[K_VALUE-3] = np.where(labelsDev == pred)[0].shape[0] * 1. / DEV_SET
		print 'Accuracy for K={}: {}'.format(K_VALUE, accuracy[K_VALUE-3])		

	print 'Best Accuracy obtained for K={}'.format(np.argmax(accuracy)+3)


def main():
	# Choose the following genres: Classical, HipHop, Jazz and Rock
	genres = np.array([1,4,5,9], dtype=np.int)
	splitData(genres)

	# Fit KNN
	# meanTrain, covTrain, labels, meanTest, covTest, labelsTest = readData()
	# knnKLD(meanTrain, covTrain, labels, meanTest, covTest, labelsTest)

if __name__ == '__main__':
	main()
