import librosa
import glob
import os
import numpy as np

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try: 
                X, sample_rate = librosa.load(fn)
                sn = fn.split('/')[-1]
                sn = sn.split('.')[0]
                np.savetxt("Soundfiles/{}/{}.txt".format(sub_dir, sn), X)
            except Exception as e:
                print("{} >> {}".format(fn, e))
            
if __name__ == "__main__":
    parent_dir = 'Soundfiles'

    sub_dirs = ['192000'] # fold1, fold4
    parse_audio_files(parent_dir,sub_dirs)