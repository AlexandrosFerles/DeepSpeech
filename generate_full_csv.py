from extract_data import extract_data
import os
import pickle
from sphfile import SPHFile
import io


def generate_full_csv(stasjon_path):
    all_wav_files_ordered = []
    wav_file_sizes = []
    text_transcriptions = []
    speakers = []
    sexes = []
    births = []
    youths = []

    cnt = 0

    for path, subdirs, files in os.walk(stasjon_path):
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



                os.system(f"mv {final_path} {final_path.replace('.wav', '_old.wav')}")
                sox_command = f"sox {final_path.replace('.wav', '_old.wav')} -r 16000 -c 1 -b 16 {final_path}"
                os.system(sox_command)

                cnt += 1
                print(cnt)
                all_wav_files_ordered.append(final_path)
                wav_file_sizes.append(os.path.getsize(final_path))
                # print(final_path)
                # print(os.path.getsize(final_path))
                speakers.append(speaker)

                text_transcriptions.append(wavs_and_transcriptions[key].lower())
                sexes.append(sex)
                births.append(region_of_birth)
                youths.append(region_of_youth)

    # with open('all_wav_files_ordered.pickle', 'wb') as  all_wav_files_ordered_pickle, \
    #         open('speakers.pickle', 'wb') as  speakers_pickle, \
    #         open('text_transcriptions.pickle', 'wb') as  text_transcriptions_pickle, \
    #         open('sexes.pickle', 'wb') as  sexes_pickle, \
    #         open('births.pickle', 'wb') as  births_pickle, \
    #         open('youths.pickle', 'wb') as  youths_pickle:
    #
    #     pickle.dump(all_wav_files_ordered, all_wav_files_ordered_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(speakers, speakers_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(text_transcriptions, text_transcriptions_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(sexes, sexes_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(births, births_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(youths, youths_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    final_dict = {"WAV_file_path": all_wav_files_ordered,
                  "WAV_file_size": wav_file_sizes,
                  "Text_Transcription": text_transcriptions}

    write_to_csv(filename='/home/guest/Desktop/Dataset16/'
                                    'sve.16khz.0467-1/0467_sv_train_1/Stasjon1_data.csv', dictionary=final_dict)





def write_to_csv(filename, dictionary):

    wav = dictionary["WAV_file_path"]
    filesizes = dictionary["WAV_file_size"]
    text = dictionary["Text_Transcription"]

    with open(f'{filename}', 'w') as csv_file:

        csv_file.write("WAV_file_path" + "," + "WAV_file_size" + "," + "Text_Transcription")
        csv_file.write("\n")

        for index, elem in enumerate(wav):
            csv_file.write(f'{elem}' + "," + f"{filesizes[index]}" + "," + f"{text[index]}")
            csv_file.write("\n")


if __name__ == '__main__':
    generate_full_csv(stasjon_path='/home/guest/Desktop/Dataset16/sve.16khz.0467-1/0467_sv_train_1/Stasjon1')
