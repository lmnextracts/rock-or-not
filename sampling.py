import os
import librosa 
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks


######################################################
# Constants
######################################################
framelen_ms = 20
dataDir = 'C:\\Users\\Lakshmi\\Desktop\\229 Project\\DataSamples'

######################################################
# Global Variables
######################################################
samplesMatrix = np.zeros((750000,21))
index = 0

def getLabel(genre):
	return {
	'blues':0,
	'classical': 1,
	'country': 2,
	'disco': 3,
	'hiphop':4,
	'jazz':5,
	'metal':6,
	'pop':7,
	'reggae':8,
	'rock':9
	}.get(genre, -1)

getLabel('blues')

for subdir,dirs,files in os.walk(dataDir):
	for file in files:
		basename = os.path.basename(subdir)
		label = getLabel(basename)

		# make 750 chunks (length: 20ms) for each sample (length: 15s)
		filePath = os.path.join(subdir,file)
		sample = AudioSegment.from_file(filePath)
		chunks = make_chunks(sample, framelen_ms)

		# make 750 x 21 matrix for each sample
		print ('Processing chunks for file ' + file + ' with label : {}').format(label)
		for i,chunk in enumerate(chunks):
			print 'Chunk No: {}'.format(i)
			chunk.export('chunk.au', format='au')
			y,sr = librosa.load('chunk.au')
			featureVec = librosa.feature.mfcc(y=y, sr=sr)
			featureVec = np.append(featureVec, label)
			samplesMatrix[index] = featureVec.reshape(1,21)
			index += 1

np.savetxt('data.csv', samplesMatrix, delimiter = ',',newline = '\n')

		

# file = 'sample.au'
# sample = AudioSegment.from_file(file)
# chunks = make_chunks(sample, framelen_ms)

# # make 750 x 21 matrix for each sample
# for chunk in chunks:
# 	chunk.export('chunk.au', format='au')
# 	y,sr = librosa.load('chunk.au')
# 	featureVec = librosa.feature.mfcc(y=y, sr=sr)
# 	featureVec = np.append(featureVec, 1)
# 	print featureVec.reshape()





