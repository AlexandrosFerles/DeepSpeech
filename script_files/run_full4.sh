#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py \
  --train_files /home/guest/Desktop/DeepSpeech/data/TRAIN/train.csv \
  --dev_files /home/guest/Desktop/DeepSpeech/data/DEV/dev.csv \
  --test_files /home/guest/Desktop/DeepSpeech/data/TEST/test.csv \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --test_batch_size 32 \
  --n_hidden 350 \
  --epoch 100 \
  --validation_step 1 \
  --early_stop True \
  --earlystop_nsteps 6 \
  --estop_mean_thresh 0.1 \
  --estop_std_thresh 0.1 \
  --dropout_rate 0.4 \
  --learning_rate 0.001 \
  --report_count 100 \
  --train True \
  --use_seq_length False \
  --export_dir /home/guest/Desktop/DeepSpeech/results/model_export/ \
  --checkpoint_dir /home/guest/Desktop/DeepSpeech/results/checkout/ \
  --alphabet_config_path /home/guest/Desktop/DeepSpeech/data/alphabet.txt\
  --lm_binary_path /home/guest/Desktop/DeepSpeech/data/lm/lm_with_family_transciptions/lm.binary \
  --lm_trie_path /home/guest/Desktop/DeepSpeech/data/lm/lm_with_family_transciptions/trie 
\ 
"$@"
