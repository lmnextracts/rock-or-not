import numpy as np
import os		

def main():
	fileName = os.path.join('C:\\Users\\Lakshmi\\Desktop\\repo229\\rock-or-not\\data', 'covTest_15.csv')
	data = np.loadtxt(fileName, delimiter = ',')
	print data.shape

if __name__ == '__main__':
	main()