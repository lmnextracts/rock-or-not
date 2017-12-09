import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display

FILEPATH = 'C:\\Users\\Lakshmi\\Desktop\\repo229\\rock-or-not\\samples'

def main():
	filePath = os.path.join('pop.00000.au')
	sample = AudioSegment.from_file(filePath)
	y,sr = librosa.load(filePath)

	plt.figure()
	C = librosa.feature.chroma_cqt(y=y, sr=sr)
	librosa.display.specshow(C, y_axis='chroma')
	plt.colorbar()
	plt.title('Chromagram')
	plt.savefig(filePath[0:-9] + '.png'	)			

	return

if __name__ == '__main__':
	main()