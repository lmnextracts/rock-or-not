from pydub import AudioSegment as aud
import os

rootDir = 'C:\\Users\\Lakshmi\\Desktop\\229 Project\\dataset_original\\genres'
outputDir = 'C:\\Users\\Lakshmi\Desktop\\229 Project\\DataSamples\\'

startMin = 0
startSec = 5

endMin = 0
endSec = 20

startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000

fileName = 'osaka.mp3'

for subdir,dirs,files in os.walk(rootDir):
	print '\n\n'
	for file in files:
		basename = os.path.basename(subdir)
		if not os.path.exists(os.path.join(outputDir,basename)):
			os.makedirs(os.path.join(outputDir,basename))
		inputPath = os.path.join(subdir,file)
		outputPath = os.path.join(outputDir,basename,file)	

		sample = aud.from_file(inputPath)
		croppedSample = sample[startTime:endTime]
		print 'Exporting ' + file + ' to ' + outputPath
		croppedSample.export(outputPath, format = 'au')



