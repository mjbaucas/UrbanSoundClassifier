import os

if __name__ == "__main__":
	file_name = "testsound.wav"
	os.system("python3 ../SoundRecorder/soundrec.py {}".format(file_name))
	os.system("python3 ../UrbanSoundClassifier/testclassifier.py {}".format(file_name))
