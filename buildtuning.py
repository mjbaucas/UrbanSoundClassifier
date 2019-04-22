import os

if __name__ == "__main__":
    for i in range(1,7):
        for j in range(1,7):
            os.system('python buildmodelB.py {} {} {}'.format(3000, i*50, j*50))