import os
import time
import socket

if __name__ == "__main__":
	file_name = "testsound.wav"
	
	time_file = open("recordtime.txt", "a+")
	
	for i in range(0, 30):
		lapse = time.time()
		time_file.write("Start {}: {}\n".format(i, lapse))
		os.system("python3 ../SoundRecorder/soundrec.py {}".format(file_name))
		lapse = time.time()
		time_file.write("Middle {}: {}\n".format(i, lapse))
		sound_file = open(file_name, "rb")
		sound_data = sound_file.read()
		client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client.connect(("10.16.10.240", 32500))

		while sound_data:
			data = sound_data[:4096]
			if not data:
				break
			sent = client.send(data)
			sound_data = sound_data[sent:]
			#print(sent)
				
		from_server = client.recv(4096)	
		print(from_server)
		lapse = time.time()
		time_file.write("End {}: {}\n".format(i, lapse))		
	time_file.close()
