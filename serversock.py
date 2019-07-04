
import socket
import os
import time
from datetime import datetime

address_list = [
    '10.11.245.212',
    '10.11.170.148',
    '10.11.246.184',
    '10.11.170.102',
    '10.11.155.199',
    '10.11.182.13',
    '10.11.251.59',
    '10.11.170.72',
    '10.11.213.153',
    '10.11.189.174',
    '10.11.241.48',
    '10.11.178.217'
]

if __name__ == "__main__":
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("10.11.148.56", 32500))
    #server.bind(("10.16.10.240", 32500))
    server.listen(5)

    id_counter = 0
    time_file = open("cloud_latency_12.txt", "a+")
    it_counter = 0
    while True:
        connection, address = server.accept()
        print(address)
        if address[0] != address_list[id_counter]:
            connection.close()
            print("Skip")
        else:
            start = datetime.now().timestamp()
            time_file.write("Start {} {}\n".format(address[0], start))
            file_name = "testsound_npy.txt"
            size_count = 0
            with open(file_name, "wb") as f:
                while True:
                    recieve = connection.recv(2048)
                    size_count += len(recieve)
                    if not recieve or recieve == "" or size_count > 312000:
                    #if not recieve or recieve == "" or size_count > 1600:
                    #if not recieve or recieve == "" or size_count > 0:
                        f.write(recieve)
                        break
                    f.write(recieve)
            end = datetime.now().timestamp()
            time_file.write("End {} {}\n".format(address[0], end))
            #os.system("python3 testmodel.py {}".format(file_name))
            connection.send("Done".encode('utf-8'))
            connection.close()
            print("Done")
        
            id_counter+=1
            if id_counter >= len(address_list):
                id_counter = 0
            it_counter+=1
            print("{} {}".format(it_counter, id_counter))
            
