import numpy as np
import pandas as py

NUMBER_OF_SONGS = 1000

def main():
	# Initialize mean matrix
	mean_20 = np.zeros((1000,20))
	mean_15 = np.zeros((1000,15))

	# Initialize covariance matrix
	cov_20 = np.zeros((1000,20,20))
	cov_15 = np.zeros((1000,15,15))

	for i in xrange(0,NUMBER_OF_SONGS):
		# Read data for each song
		data_20 = py.read_csv('data.csv', header=None, index_col=False, nrows=750, skiprows=(750*i));
		data_20 = data_20.iloc[:,0:20]
		data_15 = data_20.iloc[:,0:15]

		mean_20[i] = data_20.mean().as_matrix().T
		mean_15[i] = data_15.mean().as_matrix().T

		cov_20[i] = data_20.cov().as_matrix()
		cov_15[i] = data_15.cov().as_matrix()

	cov_20 = cov_20.reshape(NUMBER_OF_SONGS*20,20)
	cov_15 = cov_15.reshape(NUMBER_OF_SONGS*15,15)

	mean_20 = py.DataFrame(mean_20)
	mean_15 = py.DataFrame(mean_15)
	cov_20 = py.DataFrame(cov_20)
	cov_15 = py.DataFrame(cov_15)


	mean_20.to_csv('mean_20.csv', header=False, index=False)
	mean_15.to_csv('mean_15.csv', header=False, index=False)
	cov_20.to_csv('cov_20.csv', header=False, index=False)
	cov_15.to_csv('cov_15.csv', header=False, index=False)



if __name__ == '__main__':
	main()
