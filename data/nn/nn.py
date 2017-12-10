import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

# user-defined constants
GENRES = np.array([1,4,5,7], dtype = np.int)
NUM_GENRES = len(GENRES)
K_START = 3
K_END = 201
FEATURE_DIM = 15
ITERATIONS = 10

DATASET_SIZE = 400
TRACK_COUNT_PER_GENRE = 100
CURRENT_DATASET_SIZE = NUM_GENRES * TRACK_COUNT_PER_GENRE
TRAINSET_PERCENT = 0.7
TESTSET_PERCENT = 0.3

TRAIN_SET = int(TRAINSET_PERCENT * CURRENT_DATASET_SIZE)
TEST_SET = int(TESTSET_PERCENT * CURRENT_DATASET_SIZE)

FILEPATH_DATA = 'D:\\repo\\rock-or-not\\data\\nn'
FILEPATH_PLOTS = 'D:\\repo\\rock-or-not\\plots\\nn'


FILEPATH_MEAN = os.path.join(FILEPATH_DATA,'mean_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV = os.path.join(FILEPATH_DATA, 'cov_{}.csv'.format(FEATURE_DIM))
FILEPATH_LABELS = os.path.join(FILEPATH_DATA, 'labels.csv')

FILEPATH_MEAN_TRAIN = os.path.join(FILEPATH_DATA, 'meanTrain_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV_TRAIN = os.path.join(FILEPATH_DATA, 'covTrain_{}.csv'.format(FEATURE_DIM))
FILEPATH_MEAN_TEST = os.path.join(FILEPATH_DATA, 'meanTest_{}.csv'.format(FEATURE_DIM))
FILEPATH_COV_TEST = os.path.join(FILEPATH_DATA, 'covTest_{}.csv'.format(FEATURE_DIM))
FILEPATH_LABELS_TRAIN = os.path.join(FILEPATH_DATA, 'labelsTrain.csv')
FILEPATH_LABELS_TEST = os.path.join(FILEPATH_DATA, 'labelsTest.csv')

def getGenre(label):
	return {
	0:'blues',
	1:'classical',
	2:'country',
	3:'disco',
	4:'hiphop',
	5:'jazz',
	6:'metal',
	7:'pop',
	8:'reggae',
	9:'rock',
	}.get(label, 'Unknown')

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

	# Save split data as CSV files
	np.savetxt('neuralMean.csv', mean, delimiter=',', newline='\n')
	np.savetxt('neuralCov.csv', cov.reshape(len(indices),FEATURE_DIM*FEATURE_DIM), delimiter=',', newline='\n')
	np.savetxt('neuralLabels.csv', labels, delimiter=',',newline='\n')
	return

def splitTrainTest():
	data = np.loadtxt('dataMatrix.csv', delimiter=',')
	labels = np.loadtxt('neuralLabels.csv', delimiter =',')

	idxTrain = np.zeros(280)
	idxDev = np.zeros(60)
	idxTest = np.zeros(60)
	for x in xrange(0,4):
		a = range(x * 100, x*100 + 70)
		idxTrain = np.append(idxTrain,a)
		a = range(x*100 + 70, x*100 + 85)
		idxDev = np.append(idxDev,a)
		a = range(x*100 + 85, x*100 + 100)
		idxTest = np.append(idxTest,a)

	# Extract train and test records and save as CSV
	idxTrain = np.array(idxTrain, dtype = np.int)
	idxDev = np.array(idxDev, dtype = np.int)
	idxTest = np.array(idxTest, dtype = np.int)


	# Save split data as CSV files
	np.savetxt('dataTrain.csv', data[idxTrain],delimiter=',',newline='\n')
	np.savetxt('labelsTrain.csv', labels[idxTrain],delimiter=',',newline='\n')
	np.savetxt('dataDev.csv', data[idxDev],delimiter=',',newline='\n')
	np.savetxt('labelsDev.csv', labels[idxDev],delimiter=',',newline='\n')
	np.savetxt('dataTest.csv', data[idxTest],delimiter=',',newline='\n')
	np.savetxt('labelsTest.csv', labels[idxTest],delimiter=',',newline='\n')
	return

def convertToFeatureMatrix():
	meanData = np.loadtxt('neuralMean.csv', delimiter=',')
	covData = np.loadtxt('neuralCov.csv', delimiter=',')

	print meanData.shape
	print covData.shape

	covData.shape = (400,15,15)

	upperTriangular = np.triu_indices(15)
	cov = np.zeros((400,120))
	for x in xrange(0,400):
		cov[x] = covData[x][upperTriangular]
	print cov.shape

	cov.shape = (400,120)

	data = np.zeros((400,135))	
	data[:,0:15] = meanData
	data[:,15:] = cov

	np.savetxt('dataMatrix.csv', data, delimiter=',', newline='\n')
	return

def buildNN():
	dataTrain = np.loadtxt('dataTrain.csv', delimiter=',')
	labelsTrain = np.loadtxt('labelsTrain.csv', delimiter=',')

	dataDev = np.loadtxt('dataDev.csv', delimiter=',')
	labelsDev = np.loadtxt('labelsDev.csv', delimiter=',')

	dataTest = np.loadtxt('dataTest.csv', delimiter=',')
	labelsTest = np.loadtxt('labelsTest.csv', delimiter=',')

	accuracy = np.zeros((198,198))
	for x in xrange(2,200):
		for y in xrange(2,x+1):			
			nnModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(x,y), random_state=1)
			nnModel.fit(dataTrain,labelsTrain)
			accuracy[x-2][y-2] = nnModel.score(dataDev,labelsDev)					
			print '[{},{}] Accuracy: {}'.format(x,y,accuracy[x-2][y-2])

	idx = np.argmax(accuracy)
	max_X = int(idx / 198)
	max_Y = int(idx % 198)
	print 'Max. Accuracy for ({},{}) neural network with Accuracy={}.'.format(max_X, max_Y, np.max(accuracy))

	np.savetxt('accuracy.csv',accuracy, delimiter=',',newline='\n')
	return

def findBestPCAComponent():
	dataTrain = np.loadtxt('dataTrain.csv', delimiter=',')
	labelsTrain = np.loadtxt('labelsTrain.csv', delimiter=',')

	dataDev = np.loadtxt('dataDev.csv', delimiter=',')
	labelsDev = np.loadtxt('labelsDev.csv', delimiter=',')

	dataTest = np.loadtxt('dataTest.csv', delimiter=',')
	labelsTest = np.loadtxt('labelsTest.csv', delimiter=',')

	data = np.append(dataTrain,dataDev,axis=0)
	labels = np.append(labelsTrain,labelsDev,axis=0)

	rangeBegin = 20
	rangeEnd = 135
	score = np.zeros(rangeEnd-rangeBegin)
	for p in xrange(rangeBegin,rangeEnd):
		pca = PCA(n_components=p)
		pcaDataTrain = pca.fit_transform(data)
		pcaDataTest = pca.fit_transform(dataTest)
		print pcaDataTest.shape
		kmeans = KMeans(n_clusters=4, random_state=0).fit(pcaDataTrain)
		score[p-rangeBegin] = normalized_mutual_info_score(labelsTest, kmeans.predict(pcaDataTest))
		print 'P[{}] {}'.format(p,score[p-rangeBegin])

	print 'Max. Score at P[{}] with MIS = {}'.format(np.argmax(score)+rangeBegin, np.max(score))
	return


def main():
	# splitData(GENRES)
	# splitTrainTest()
	# buildNN()
	findBestPCAComponent()
	return

if __name__ == '__main__':
	main()