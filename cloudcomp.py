import os
import time
import socket

if __name__ == "__main__":
	file_name = "testsound.wav"
	
	time_file = open("cloudtime.txt", "a+")
	
	for i in range(0, 20):
		lapse = time.time()
		time_file.write("Start {}: {}\n".format(i, lapse))
		os.system("python3 ../SoundRecorder/soundrec.py {}".format(file_name))
		
		sound_file = open(file_name, "rb")
		sound_data = sound_file.read()
		client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client.connect(("10.16.10.240", 32500))
		
		lapse = time.time()
		time_file.write("Middle {}: {}\n".format(i, lapse))
		while sound_data:
			sent = client.send(sound_data)
			if not sent:
				break
			sound_data = sound_data[sent:]
			print(sound_data)			
		lapse = time.time()
		time_file.write("End {}: {}\n".format(i, lapse))
