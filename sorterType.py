import glob
import os
import shutil
import scipy.io.wavfile as wavfile
import random
from shutil import copyfile

def sort_audio_files(parent_dir,sub_dirs, percent_split, file_ext='*.wav'):
    sound_pool = {}
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try: 
                sn = fn.split('/')[2]
                sound_type = fn.split('/')[1]
                sound_class = sn.split('-')[1]
                if sound_type + "-" + sound_class in sound_pool:
                    sound_pool[sound_type + "-" + sound_class].append(sn)
                else:
                    sound_pool[sound_type + "-" + sound_class] = [sn]
            except Exception as e:
                print("{} >> {}".format(fn, e))
    # print(sound_pool)

    for i in range(0,10):
        div_dir = "SoundfilesB/{}".format(i)
        if not os.path.exists(div_dir):
            os.makedirs(div_dir)
    
    for key, items in sound_pool.items():
        for item in items:
            key_class = key.split('-')[0]
            key_type = key.split('-')[1]

            sound_split = item.split('.')[0]
            div_dir = "SoundfilesB/{}".format(key)
            if not os.path.isfile("{}/{}.txt".format(div_dir, sound_split)):
                try:
                    copyfile("{}/{}/{}.txt".format(parent_dir, key_class, sound_split), "{}/{}/{}.txt".format('SoundfilesB', key_type, sound_split))
                    #print("{}/{}/{}.txt".format(parent_dir, key_split, sound_split))
                except Exception as e:
                    print('{}'.format(e))
if __name__ == "__main__":
    parent_dir = 'Soundfiles'

    sub_dirs = ['8000', '11025', '16000', '22050', '24000', '36000', '44100', '48000', '96000', '192000'] # fold1, fold4
    sort_audio_files(parent_dir,sub_dirs, 100)