from extract_data import extract_data
import os
import pickle
from sphfile import SPHFile
import io
import subprocess
import string
from threading import RLock
import soundfile as sf
from collections import Counter
import random
from util.audio import audiofile_to_input_vector
from util.text import text_to_char_array, Alphabet
import numpy as np


SAMPLE_RATE = 16000

training_percent = 0.9
validation_percent = 0.1
# test_percent = 0.05

numcontext = 9
numcep = 26

alphabet = Alphabet(os.path.abspath('/home/guest/Desktop/DeepSpeech/data/alphabet.txt'))


excluded_train_wavs = ['/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon5/060799_a/adb_0467/speech/scr0467/05/04670504/r4670396/u0396196.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon5/220799/adb_0467/speech/scr0467/05/04670505/r4670441/u0441079.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon5/280799/adb_0467/speech/scr0467/05/04670505/r4670451/u0451201.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/160799/adb_0467/speech/scr0467/07/04670706/r4670598/u0598036.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/160799/adb_0467/speech/scr0467/07/04670706/r4670598/u0598037.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/160799/adb_0467/speech/scr0467/07/04670706/r4670598/u0598102.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672173.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672174.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672175.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672176.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672177.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672178.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672179.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672180.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672181.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672182.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672183.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672184.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672185.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672186.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672187.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672188.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672189.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672190.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672191.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672192.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672193.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672194.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672195.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672196.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672197.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672198.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672199.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672200.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672201.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672202.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672203.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672204.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672206.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672213.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672214.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672215.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672216.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672217.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672218.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672219.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672220.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672221.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672222.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672223.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100899/adb_0467/speech/scr0467/07/04670707/r4670672/u0672224.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/100799/adb_0467/speech/scr0467/07/04670706/r4670580/u0580189.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-2/0467_sv_train_2/Stasjon7/060899/adb_0467/speech/scr0467/07/04670707/r4670658/u0658012.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon9/050899/adb_0467/speech/scr0467/09/04670909/r4670889/u0889044.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon9/050899/adb_0467/speech/scr0467/09/04670909/r4670889/u0889070.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon9/050899/adb_0467/speech/scr0467/09/04670909/r4670889/u0889111.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon9/220799/adb_0467/speech/scr0467/09/04670909/r4670862/u0862092.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon9/070899/adb_0467/speech/scr0467/09/04670910/r4670902/u0902141.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon20/041199/adb_0467/speech/scr0467/20/04672002/r4670107/u0107038.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon8/020899/adb_0467/speech/scr0467/08/04670808/r4670756/u0756179.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-3/0467_sv_train_3/Stasjon8/040899/adb_0467/speech/scr0467/08/04670808/r4670762/u0762009.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-1/0467_sv_train_1/Stasjon2/160799/adb_0467/speech/scr0467/02/04670202/r4670137/u0137149.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-1/0467_sv_train_1/Stasjon3/200799/adb_0467/speech/scr0467/03/04670303/r4670245/u0245096.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-1/0467_sv_train_1/Stasjon1/160799/adb_0467/speech/scr0467/01/04670101/r4670036/u0036190.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-1/0467_sv_train_1/Stasjon1/030899/adb_0467/speech/scr0467/01/04670101/r4670081/u0081161.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-1/0467_sv_train_1/Stasjon1/190799/adb_0467/speech/scr0467/01/04670101/r4670041/u0041027.wav',
'/home/guest/Desktop/Dataset16/TrainSet/sve.16khz.0467-1/0467_sv_train_1/Stasjon1/190799/adb_0467/speech/scr0467/01/04670101/r4670036/u0036190.wav',]



def convert_wav(path):
    cnt = 0
    for path, subdirs, files in os.walk(path):
        for name in files:
            if not name.endswith('wav'):
                continue

            final_path = os.path.join(path, name)

            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)

            os.system(f"mv {final_path} {final_path.replace('.wav', '_old.wav')}")
            sox_command = f"sox {final_path.replace('.wav', '_old.wav')} -r 16000 -c 1 -b 16 {final_path}"
            os.system(sox_command)

            os.system(f"rm {final_path.replace('.wav', '_old.wav')}")


def generate_statistics(path):
    all_wav_files_ordered = []
    wav_file_sizes = []
    text_transcriptions = []
    speakers = []
    sexes = []
    births = []
    youths = []

    cnt = 0
    counter_short_long = 0
    for path, subdirs, files in os.walk(path):
        for name in files:
            if not name.endswith('spl'):
                continue

            spl_path = os.path.join(path, name)
            speaker, name, sex, region_of_birth, region_of_youth, wavs_and_transcriptions = extract_data(spl_path)



            wav_path = spl_path.replace('.spl', '/').replace('data', 'speech')

            for key in wavs_and_transcriptions:

                final_path = f'{wav_path}{key}'

                if not os.path.isfile(final_path):
                    continue

                f = sf.SoundFile(final_path)
                duration = len(f) / f.samplerate
                #

                # if duration < 2:
                #     counter_short_long += 1
                #     continue

                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt)

                all_wav_files_ordered.append(final_path)
                wav_file_sizes.append(os.path.getsize(final_path))

                speakers.append(speaker)

                lower_text = wavs_and_transcriptions[key].lower() \
                    .replace(',', ' ').replace('è', "e").replace('é', "e").replace("ÿ", "y").replace("ü", "u")

                for c in string.punctuation:
                    lower_text = lower_text.replace(c, " ")

                text_transcriptions.append(lower_text)
                sexes.append(sex)
                births.append(region_of_birth)
                youths.append(region_of_youth)

    group_sexes = Counter(sexes)
    group_births = Counter(births)
    group_youths = Counter(youths)
    group_speakers = Counter(speakers)

    print('=========SEXES==========')
    print(group_sexes.values())
    print(group_sexes.keys())
    print('===================')

    print('=========births==========')
    print(group_births.values())
    print(group_births.keys())
    print('===================')

    print('=========youths==========')
    print(group_youths.values())
    print(group_youths.keys())
    print('===================')

    print('=========speakers==========')
    print(len(group_speakers.values()))
    print(group_speakers.values())
    print(group_speakers.keys())
    print('===================')

    print(counter_short_long)


def generate_statistics_from_file(path, wavs):
    wav_paths_read = []
    speakers = []
    sexes = []
    births = []
    youths = []

    cnt = 0
    for path, subdirs, files in os.walk(path):
        for name in files:
            if not name.endswith('spl'):
                continue

            spl_path = os.path.join(path, name)
            speaker, name, sex, region_of_birth, region_of_youth, wavs_and_transcriptions = extract_data(spl_path)

            wav_path = spl_path.replace('.spl', '/').replace('data', 'speech')

            for key in wavs_and_transcriptions:

                final_path = f'{wav_path}{key}'

                # if final_path not in wavs:
                #     continue

                cnt += 1
                if cnt % 100000 == 0:
                    print(cnt)

                speakers.append(speaker)
                wav_paths_read.append(final_path)
                sexes.append(sex)
                births.append(region_of_birth)
                youths.append(region_of_youth)


    indices = []
    for path in wavs:
        indices.append(wav_paths_read.index(path))
    # indices = int(indices)

    group_sexes = Counter([sexes[i] for i in indices])
    group_births = Counter([births[i] for i in indices])
    group_youths = Counter([youths[i] for i in indices])
    group_speakers = Counter([speakers[i] for i in indices])

    print('=========SEXES==========')
    print(group_sexes.values())
    print(group_sexes.keys())
    print('===================')

    print('=========births==========')
    print(group_births.values())
    print(group_births.keys())
    print('===================')

    print('=========youths==========')
    print(group_youths.values())
    print(group_youths.keys())
    print('===================')

    print('=========speakers==========')
    print(group_speakers.values())
    print(group_speakers.keys())
    print('===================')


def generate_statistics_csv(filename, data_path):
    wavs = read_csv(filename)

    generate_statistics_from_file(data_path, wavs)


def generate_full_csv(path, write_file):
    all_wav_files_ordered = []
    wav_file_sizes = []
    text_transcriptions = []
    speakers = []
    sexes = []
    births = []
    youths = []

    cnt = 0
    counter_short_long = 0
    for path, subdirs, files in os.walk(path):
        for name in files:

            # if len(all_wav_files_ordered) > 20000:
            #     break

            if not name.endswith('spl'):
                continue

            spl_path = os.path.join(path, name)
            speaker, name, sex, region_of_birth, region_of_youth, wavs_and_transcriptions = extract_data(spl_path)

            if not sex:
                continue

            wav_path = spl_path.replace('.spl', '/').replace('data', 'speech')

            for key in wavs_and_transcriptions:

                final_path = f'{wav_path}{key}'

                if not os.path.isfile(final_path):
                    continue

                if final_path in excluded_train_wavs:
                    print(final_path)
                    continue


                # f = sf.SoundFile(final_path)
                # duration = len(f) / f.samplerate

                # if duration < 4:
                #     counter_short_long+=1
                #     continue

                cnt += 1
                if cnt % 10000 == 0:
                    print(cnt)
                # if train then those produce inf in loss
                #if cnt >= 60000 and cnt <= 70000:
                #    print(cnt)
                #    continue

                #if cnt >= 110000 and cnt <= 150000:
                #     print(cnt)
                #     continue
                #if test the those produce inf loss
                if cnt > 15000 and cnt < 19000:
                    continue

                lower_text = wavs_and_transcriptions[key].lower() \
                    .replace(',', ' ').replace('è', "e").replace('é', "e").replace("ÿ", "y").replace("ü", "u").replace(
                    'î', 'i')

                for c in string.punctuation:
                    lower_text = lower_text.replace(c, " ")

                #features = audiofile_to_input_vector(final_path, numcep, numcontext)
                #features_len = len(features) - 2 * numcontext
                #transcript = text_to_char_array(lower_text, alphabet)

                #if features_len < len(transcript):
                #    counter_short_long += 1
                #    print(final_path)
                #    continue

                speakers.append(speaker)

                all_wav_files_ordered.append(final_path)
                wav_file_sizes.append(os.path.getsize(final_path))

                text_transcriptions.append(lower_text)
                sexes.append(sex)
                births.append(region_of_birth)
                youths.append(region_of_youth)

    group_births = Counter(births)


    # num_test_data = int(test_percent * len(all_wav_files_ordered))

    train_wav_indices = []
    dev_wav_indices = []

    for region in group_births.keys():
        # dict_with_region_indices[region] = np.where(group_births == region)[0]
        indices = np.where(np.array(births) == region)[0]

        #if 'Stockholm' in region:
        #    indices = indices[0:20000]
        #elif 'Göteborg' in region:
        #    indices = indices[0:10000]
        #else:
        indices = indices[0:5000]

        num_train_data = int(training_percent * len(indices))

        train_indices = np.array(indices[0:num_train_data])
        dev_indices = np.array(indices[num_train_data:])

        train_wav_indices.extend([i for i in train_indices])
        dev_wav_indices.extend([i for i in dev_indices])


    train_wav_files = [all_wav_files_ordered[i] for i in train_wav_indices]
    dev_wav_files = [all_wav_files_ordered[i] for i in dev_wav_indices]
    # test_wav_files = all_wav_files_ordered[num_train_data + num_dev_data:]
    #
    train_wav_file_sizes = [wav_file_sizes[i] for i in train_wav_indices]
    dev_wav_file_sizes = [wav_file_sizes[i] for i in dev_wav_indices]
    # test_wav_file_sizes = wav_file_sizes[num_train_data + num_dev_data:]
    #
    train_transcriptions_files = [text_transcriptions[i] for i in train_wav_indices]
    dev_transcriptions_files = [text_transcriptions[i] for i in dev_wav_indices]
    # test_transcriptions_files = text_transcriptions[num_train_data + num_dev_data:]

    train_dict = {"wav_filename": train_wav_files,
                  "wav_filesize": train_wav_file_sizes,
                  "transcript": train_transcriptions_files}

    val_dict = {"wav_filename": dev_wav_files,
                  "wav_filesize": dev_wav_file_sizes,
                  "transcript": dev_transcriptions_files}

    # test_dict = {"wav_filename": test_wav_files,
    #               "wav_filesize": test_wav_file_sizes,
    #               "transcript": test_transcriptions_files}
    #
    #write_to_csv(filename='/home/guest/Desktop/DeepSpeech/data/TRAIN/train.csv', dictionary=train_dict)

    #write_to_csv(filename='/home/guest/Desktop/DeepSpeech/data/DEV/dev.csv', dictionary=val_dict)
    #
    # write_to_csv(filename='/home/guest/Desktop/DeepSpeech/data/TEST/test.csv', dictionary=test_dict)

    # unique_transcr = set(text_transcriptions)
    # list_ = list(unique_transcr)
    #
    # print('text : ',len(text_transcriptions))
    # print('unique : ',len(list_))
    #
    # final_dict = {"transcript": list_}
    # write_to_csv_2(filename='/home/guest/Desktop/Dataset16/sve.16khz.0467-1/0467_sv_train_1/stah2.csv', dictionary=final_dict)

    final_dict = {"wav_filename": all_wav_files_ordered,
                  "wav_filesize": wav_file_sizes,
                  "transcript": text_transcriptions}

    write_to_csv(filename=write_file, dictionary=train_dict)


    print('============ALL=============')


    group_sexes = Counter(sexes)
    group_births = Counter(births)
    group_youths = Counter(youths)

    print('=========SEXES==========')
    for sex in group_sexes.keys():
        print("key : " + sex + 'with value : ' + str(group_sexes[sex]))
    # print(group_sexes.values())
    # print(group_sexes.keys())
    print('===================')

    print('=========births==========')
    for sex in group_births.keys():
        print("key : " + sex + 'with value : ' + str(group_births[sex]))
    # print(group_births.values())
    # print(group_births.keys())
    print('===================')

    print('=========youths==========')
    for sex in group_youths.keys():
        print("key : " + sex + 'with value : ' + str(group_youths[sex]))
    # print(group_births.values())
    # print(group_youths.values())
    # print(group_youths.keys())
    print('===================')




    print(counter_short_long)
    print('============TRAIN=============')

    group_sexes = Counter([sexes[i] for i in train_wav_indices])
    group_births = Counter([births[i] for i in train_wav_indices])
    group_youths = Counter([youths[i] for i in train_wav_indices])

    print('=========SEXES==========')
    for sex in group_sexes.keys():
        print("key : " + sex + 'with value : ' + str(group_sexes[sex]))
    # print(group_sexes.values())
    # print(group_sexes.keys())
    print('===================')

    print('=========births==========')
    for sex in group_births.keys():
        print("key : " + sex + 'with value : ' + str(group_births[sex]))
    # print(group_births.values())
    # print(group_births.keys())
    print('===================')

    print('=========youths==========')
    for sex in group_youths.keys():
        print("key : " + sex + 'with value : ' + str(group_youths[sex]))
    # print(group_births.values())
    # print(group_youths.values())
    # print(group_youths.keys())
    print('===================')


    print('============DEV=============')

    group_sexes = Counter([sexes[i] for i in dev_wav_indices])
    group_births = Counter([births[i] for i in dev_wav_indices])
    group_youths = Counter([youths[i] for i in dev_wav_indices])

    print('=========SEXES==========')
    for sex in group_sexes.keys():
        print("key : " + sex + 'with value : ' + str(group_sexes[sex]))
    # print(group_sexes.values())
    # print(group_sexes.keys())
    print('===================')

    print('=========births==========')
    for sex in group_births.keys():
        print("key : " + sex + 'with value : ' + str(group_births[sex]))
    # print(group_births.values())
    # print(group_births.keys())
    print('===================')

    print('=========youths==========')
    for sex in group_youths.keys():
        print("key : " + sex + 'with value : ' + str(group_youths[sex]))
    # print(group_births.values())
    # print(group_youths.values())
    # print(group_youths.keys())
    print('===================')


def write_to_csv_transcripts(filename, dictionary):
    wav = dictionary["transcript"]

    with open(f'{filename}', 'w') as csv_file:
        csv_file.write("transcript")
        csv_file.write("\n")

        for index, elem in enumerate(wav):
            csv_file.write(f'{elem}')
            csv_file.write("\n")


def write_to_csv(filename, dictionary):
    wav = dictionary["wav_filename"]
    filesizes = dictionary["wav_filesize"]
    text = dictionary["transcript"]

    with open(f'{filename}', 'w', encoding='utf-8') as csv_file:
        csv_file.write("wav_filename" + "," + "wav_filesize" + "," + "transcript")
        csv_file.write("\n")

        for index, elem in enumerate(wav):
            csv_file.write(f'{elem}' + "," + f"{filesizes[index]}" + "," + f"{text[index]}")
            csv_file.write("\n")


def read_csv(filename):
    import csv

    wav_files = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                wav_files.append(row[0])
                line_count += 1
        print(f'Processed {line_count} lines.')

    return wav_files


if __name__ == '__main__':
    generate_full_csv(path='/home/guest/Desktop/Dataset16/TestSet', write_file='/home/guest/Desktop/DeepSpeech/data/TEST/test.csv')

    #generate_full_csv(path='/home/guest/Desktop/Dataset16/TrainSet', write_file='/home/guest/Desktop/Dataset16/t.csv')

    # generate_statistics(path='/home/guest/Desktop/Dataset16/TestSet')

    # generate_statistics_csv(filename='/home/guest/Desktop/Dataset16/test_mini2.csv',
    #                         data_path='/home/guest/Desktop/Dataset16/TestSet')
    # generate_statistics_csv(filename='/home/guest/Desktop/Dataset16/train_mini3.csv',
    #                         data_path='/home/guest/Desktop/Dataset16/TrainSet')
    # generate_statistics_csv(filename='/home/guest/Desktop/Dataset16/dev_mini2.csv',
    #                         data_path='/home/guest/Desktop/Dataset16/TrainSet')

    # convert_wav(path='/home/guest/Desktop/Dataset16/TestSet')
