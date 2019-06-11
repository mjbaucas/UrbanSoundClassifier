import os
import time

if __name__ == "__main__":
	file_name = "testsound.wav"
	
	time_file = open("cloudtime.txt", "a+")
	
	elapse = time.time()
	for i in range(0, 100):
		elapse = time.time() - elapse
		time_file.write("Start {}: {}".format(i, elapse))
		os.system("python3 ../SoundRecorder/soundrec.py {}".format(file_name))
		elapse = time.time() - elapse
		time_file.write("Middle {}: {}".format(i, elapse))
		os.system("python3 ../UrbanSoundClassifier/testclassifier.py {} {} {} {}".format(file_name, 5000, 280, 300))
		elapse = time.time() - elapse
		time_file.write("End {}: {}".format(i, elapse))
