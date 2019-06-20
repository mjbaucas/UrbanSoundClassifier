import os
import time
import socket

if __name__ == "__main__":
	file_name = "testsound.wav"
	
	time_file = open("cloudtime.txt", "a+")
	
	for i in range(0, 30):
		lapse = time.time()
		time_file.write("Start {}: {}\n".format(i, lapse))
		os.system("python3 ../SoundRecorder/soundrec.py {}".format(file_name))
		
		lapse = time.time()
		time_file.write("Middle {}: {}\n".format(i, lapse))
		sound_file = open(file_name, "rb")
		sound_data = sound_file.read()
		
		#Turn ethernet port on
		os.system('sudo ip link set eth0 up')
		
		client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		
		connected = False
		while connected == False:
			try:
				client.connect(("10.16.10.240", 32500))
				connected = True
			except Exception as e:
				connected = False
				time.sleep(2)
				# Do nothing
		
		while sound_data:
			data = sound_data[:4096]
			if not data:
				break
			sent = client.send(data)
			sound_data = sound_data[sent:]
				
		from_server = client.recv(4096)	
		print(from_server)
		
		#Turn ethernet port off
		os.system('sudo ip link set eth0 down')
		
		lapse = time.time()
		time_file.write("End {}: {}\n".format(i, lapse))
