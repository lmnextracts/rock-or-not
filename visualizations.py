import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pandas.tools.plotting import parallel_coordinates

# constants
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


def main():
	data = pd.read_csv(FILEPATH_MEAN, delimiter =',', header = None)
	print data.as_matrix().shape
	return

if __name__ == '__main__':
	main()

