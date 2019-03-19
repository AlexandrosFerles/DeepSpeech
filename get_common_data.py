import os
from generate_full_csv import write_to_csv
import soundfile as sf


def get_common_data(dataset_paths):
    for path in dataset_paths:
        counter_short_long = 0
        sizes = []
        filenames = []
        transcriptions = []

        for filename in os.listdir(path):
            final_path = f'{path}/{filename}'

            if 'wav' in final_path:

                f = sf.SoundFile(final_path)
                duration = len(f) / f.samplerate


                if duration < 2:
                    counter_short_long += 1
                    continue

                filenames.append(final_path)

                size = os.path.getsize(final_path)
                sizes.append(size)
                with open(final_path.replace('wav', 'wrd'), 'r') as temp:
                    for line in temp:
                        transcriptions.append(line)

        csv_filename = f'{path}.csv'
        dict = {"wav_filename": filenames,
                "wav_filesize": sizes,
                "transcript": transcriptions}
        write_to_csv(csv_filename, dictionary=dict)

        print(f'removed {counter_short_long} wav files in {path}')

if __name__ == '__main__':
    train_path = '/home/guest/Desktop/Common_data/train-clean'
    dev_path = '/home/guest/Desktop/Common_data/dev-clean'
    test_path = '/home/guest/Desktop/Common_data/test-clean'

    dataset_paths = [train_path, dev_path, test_path]

    get_common_data(dataset_paths=dataset_paths)





