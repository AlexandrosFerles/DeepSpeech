import os

def extract_data(absolute_file_path):
    """
    :return:    The attributes of the person (id, name, sex, region) along with a
                dictionary mapping a wav file with its corresponding text transcription.
    """

    skip= True
    attributes_found = False
    start = False
    delimiter = '>-<'
    wavs_and_transcriptions = dict()

    speaker = "Unknown"
    name = "Unknown"
    sex = "Unknown"
    region_of_birth = "Unknown"
    region_of_youth = "Unknown"

    for line in open(absolute_file_path, 'r', encoding='latin1'):

        if 'Validation' in line:
            break



        if not attributes_found:
            if 'Speaker' in line:
                speaker = line.split(delimiter)[1]
            elif 'Name' in line:
                name = line.split(delimiter)[1]
            elif 'Sex' in line:
                sex = line.split(delimiter)[1]
            elif 'Region of Birth' in line:
                region_of_birth = line.split(delimiter)[1]
            elif 'Region of Youth' in line:
                region_of_youth = line.split(delimiter)[1]
                attributes_found = True

        else:

            if not start:
                if '[Record states]' in line:
                    start = True
                    continue
                else:
                    continue

            else:
                if '=' in line:
                    if skip:
                        skip = False
                        continue
                    else:

                        lst = line.split(delimiter)

                        for elem in lst:
                            if 'wav' in elem:
                                wavs_and_transcriptions[elem] = lst[2]

    return speaker, name, sex, region_of_birth, region_of_youth, wavs_and_transcriptions

def write_to_csv(absolute_file_path):

    import csv

    speaker, name, sex, region_of_birth, region_of_youth, wavs_and_transcriptions = extract_data(absolute_file_path=absolute_file_path)

    filename = f'{speaker}_{name}_{sex}_{region_of_birth}_{region_of_youth}'

    wavs = []
    transcripts = []

    for key in wavs_and_transcriptions:

        wavs.append(key)
        transcripts.append(wavs_and_transcriptions[key])

    final_dict = {"WAV_file_path": wavs, "Text_Transcription": transcripts}

    with open(f'{filename}.csv', 'a', newline='') as csvfile:
        fieldnames = final_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(final_dict)

    # with open(f'{filename}.csv', 'w') as csv_file:
    #
    #     csv_file.write("WAV_file_path" + "," + "Text_Transcription")
    #     csv_file.write("\n")
    #
    #     for index, elem in enumerate(wavs):
    #         csv_file.write(f'{elem}' + "," + f"{transcripts[index]}")
    #         csv_file.write("\n")

if __name__=='__main__':

    write_to_csv(absolute_file_path='/home/guest/Desktop/Dataset16/sve.16khz.0467-1/0467 sv train 1/Stasjon1/020899/adb_0467/data/scr0467/01/04670101/r4670001.spl')
