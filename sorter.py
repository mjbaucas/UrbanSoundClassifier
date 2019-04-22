import glob
import os
import shutil
import scipy.io.wavfile as wavfile
import random
from shutil import copyfile

def sort_audio_files(parent_dir,sub_dirs, percent_split, file_ext='*.txt'):
    training_dir = "{}/TrainingSet/{}%".format(parent_dir, str(percent_split))
    if not os.path.exists(training_dir):
         os.makedirs(training_dir)

    sound_pool = {}
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try: 
                sn = fn.split('/')[2]
                sound_type = sn.split('-')[0]
                sound_class = sn.split('-')[1]
                if sub_dir + '/' + sound_type + '-' + sound_class in sound_pool:
                    sound_pool[sub_dir + '/' + sound_type + '-' + sound_class].append(sn)
                else:
                    sound_pool[sub_dir + '/' + sound_type + '-' + sound_class] = [sn]
            except Exception as e:
                print("{} >> {}".format(fn, e))
    print(sound_pool)

    for key, item in sound_pool.items():
        selection = round(len(item)*float(percent_split/100))
        sound_chunk = random.sample(item, selection)
        randIndex = random.sample(range(len(item)), selection)
        randIndex.sort()
        sound_chunk = [item[i] for i in randIndex]
        for sound in sound_chunk:
            key_split = key.split('/')[0]
            sound_split = sound.split('.')[0]
            if not os.path.isfile("{}/{}.txt".format(training_dir, sound_split)):
                try:
                    copyfile("{}/{}/{}.txt".format(parent_dir, key_split, sound_split), "{}/{}.txt".format(training_dir, sound_split))
                    #print("{}/{}/{}.txt".format(parent_dir, key_split, sound_split))
                except Exception as e:
                    print('{}'.format(e))
if __name__ == "__main__":
    parent_dir = 'SoundfilesB'

    sub_dirs = ['3'] # fold1, fold4
    sort_audio_files(parent_dir,sub_dirs, 100)