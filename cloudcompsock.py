
import socket
import os

if __name__ == "__main__":
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("10.16.10.240", 32500))
    server.listen(5)

    while True:
        connection, address = server.accept()

        file_name = "testsound2.wav"
        with open("testsound2.wav", "wb") as f:
            while True:
                recieve = connection.recv(4096)
                if not recieve or recieve == "":
                    break
                f.write(recieve)

        os.system("python3 testclassifier.py {} {} {} {}".format(file_name, 5000, 280, 300))
        connection.send("Done".encode('utf-8'))
        connection.close()
