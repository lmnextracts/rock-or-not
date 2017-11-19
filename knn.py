import numpy as np
from numpy.linalg import det, inv

DIM = 20
K_VALUE = 5
DEV_SET = 150
TEST_SET = 150
TRAIN_SET = 700

def splitData():
	# mean = np.loadtxt('mean_15.csv', delimiter = ',')
	
	idxTrain = []
	idxTest = []
	a = []
	b = []

	for x in xrange(0,10):
		a = range(x * 100, x * 100 + 70)
		idxTrain = np.append(idxTrain,a)
		b = range(x * 100 + 70, (x+1)*100)
		idxTest = np.append(idxTest,b)

	idxTrain = np.array(idxTrain, dtype = np.int)
	idxTest = np.array(idxTest, dtype = np.int)

	np.savetxt('meanTrain_15.csv',mean[idxTrain],delimiter=',',newline='\n')
	np.savetxt('meanTest_15.csv',mean[idxTest],delimiter=',',newline='\n')

	cov = np.loadtxt('cov_15.csv', delimiter = ',')
	cov = cov.reshape(1000,15,15)

	for x in xrange(0,10):
		a = range(x * 100, x * 100 + 70)
		idxTrain = np.append(idxTrain,a)
		b = range(x * 100 + 70, (x+1)*100)
		idxTest = np.append(idxTest,b)

	idxTrain = np.array(idxTrain, dtype = np.int)
	idxTest = np.array(idxTest, dtype = np.int)
	trainCov = cov[idxTrain].reshape(idxTrain.shape[0] * 15, 15)
	testCov = cov[idxTest].reshape(idxTest.shape[0] * 15, 15)

	print trainCov.shape
	print testCov.shape

	np.savetxt('covTrain_15.csv', trainCov, delimiter=',', newline='\n')
	np.savetxt('covTest_15.csv', testCov, delimiter=',', newline='\n')	

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
	meanTrain, covTrain, labels, meanTest, covTest, labelsTest = readData()
	knnKLD(meanTrain, covTrain, labels, meanTest, covTest, labelsTest)

if __name__ == '__main__':
	main()
