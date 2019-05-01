import os

if __name__ == "__main__":
    for i in range(1,7):
        for j in range(1,7):
            #os.system('python testclassifier.py {} {} {} {}'.format("Soundfiles/Testing/DogBarking1.wav", 3000, i*50, j*50))
            os.system('python testclassifier.py {} {} {} {}'.format("testsound.wav", 6000, i*50, j*50))