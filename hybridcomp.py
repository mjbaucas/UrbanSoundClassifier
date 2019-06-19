import os
import time
import socket

if __name__ == "__main__":
	file_name = "testsound.wav"
	file_name2 = "testfeatures"
	
	time_file = open("hybridtime.txt", "a+")
	
	for i in range(0, 30):
		lapse = time.time()
		time_file.write("Start {}: {}\n".format(i, lapse))
		os.system("python3 ../SoundRecorder/soundrec.py {}".format(file_name))
		
		lapse = time.time()
		time_file.write("Middle1 {}: {}\n".format(i, lapse))
		
		os.system("python3 testmodelextract.py {} {}".format(file_name, file_name2))
		
		lapse = time.time()
		time_file.write("Middle2 {}: {}\n".format(i, lapse))
		
		feature_file = open("{}.npy".format(file_name2), "rb")
		feature_data = feature_file.read()
		client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client.connect(("10.16.10.240", 32500))

		while feature_data:
			data = feature_data[:4096]
			if not data:
				break
			sent = client.send(data)
			print(sent)
			feature_data = feature_data[sent:]
				
		from_server = client.recv(4096)	
		print(from_server)
		
		lapse = time.time()
		time_file.write("End {}: {}\n".format(i, lapse))
