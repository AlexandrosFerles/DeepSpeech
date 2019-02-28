from extract_data import extract_data
import os

def generate_full_csv(stasjon_path):

    all_wav_files = []
    total_dict = {}

    for sub_folder_1 in os.listdir(stasjon_path):

        for sub_folder_2 in os.listdir(f'{stasjon_path}/{sub_folder_1}'):

            for sub_folder_3 in os.listdir(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}'):

                if 'speech' in sub_folder_3:

                    for sub_folder_4 in os.listdir(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}'):

                        for sub_folder_5 in os.listdir(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}'):

                            for sub_folder_6 in os.listdir(
                                f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}'):

                                for sub_folder_7 in os.listdir(
                                        f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}/{sub_folder_6}'):

                                    for filename in os.listdir(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}/{sub_folder_6}/{sub_folder_7}'):

                                        if '.wav' in filename:

                                            all_wav_files.append(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}/{sub_folder_6}/{sub_folder_7}/{filename}')

                if 'data' in sub_folder_3:

                        for sub_folder_4 in os.listdir(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}'):

                            for sub_folder_5 in os.listdir(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}'):

                                for sub_folder_6 in os.listdir(
                                    f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}'):

                                    for sub_folder_7 in os.listdir(
                                            f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}/{sub_folder_6}'):

                                        for filename in os.listdir(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}/{sub_folder_6}/{sub_folder_7}'):

                                            if '.wav' in filename:

                                                speaker, name, sex, region_of_birth, region_of_youth, wavs_and_transcriptions = extract_data(f'{stasjon_path}/{sub_folder_1}/{sub_folder_2}/{sub_folder_3}/{sub_folder_4}/{sub_folder_5}/{sub_folder_6}/{sub_folder_7}/{filename}')

                                                for key in wavs_and_transcriptions:





    print(all_wav_files)


if __name__=='__main__':

    generate_full_csv(stasjon_path='/home/guest/Desktop/Dataset16/sve.16khz.0467-1/0467 sv train 1/Stasjon1')