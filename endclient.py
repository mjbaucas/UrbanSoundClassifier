import os
import time
import socket

if __name__ == "__main__":	
	exit_client = False
	while exit_client == False:	
		file_name = "testsound.wav"

		sound_file = open(file_name, "rb")
		sound_data = sound_file.read()
		
		connected = False
		while connected == False:
			try:
				print("Connecting to server")
				client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				client.connect(("10.11.148.56", 32500))
				connected = True
			except Exception as e:
				time.sleep(2)
				connected = False
				# Do nothing
		
		while sound_data:
			data = sound_data[:2048]
			if not data:
				break
			try:
				sent = client.send(data)
				sound_data = sound_data[sent:]
			except Exception as e:
				time.sleep(2)  
				break
			
		from_server = client.recv(1024)

	
