import numpy as np

def readTrainingData():
	mean = np.loadtxt('mean_15.csv', delimiter = ',')
	
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

	return mean

def main():
	meanVectors = readTrainingData()


if __name__ == '__main__':
	main()
