import os
import time

if __name__ == "__main__":
	file_name = "testsound.wav"
	
	time_file = open("edgetime.txt", "a+")
	
	for i in range(0, 30):
		lapse = time.time()
		time_file.write("Start {}: {}\n".format(i, lapse))
		os.system("python3 ../SoundRecorder/soundrec.py {}".format(file_name))
		lapse = time.time()
		time_file.write("Middle {}: {}\n".format(i, lapse))
		os.system("python3 ../UrbanSoundClassifier/testmodel.py {}".format(file_name))
		lapse = time.time()
		time_file.write("End {}: {}\n".format(i, lapse))
		
	time_file.close()
