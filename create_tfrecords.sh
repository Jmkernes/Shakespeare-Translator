#!/bin/bash

echo "==== Setting up configurations ===="

INPUT_TEXT=data_clean/modern.txt
TARGET_TEXT=data_clean/original.txt
TRAIN_INP_TOKENIZER=True
TRAIN_TAR_TOKENIZER=True
INPUT_VOCAB_SIZE=13000
TARGET_VOCAB_SIZE=13000
INP_SP_MODEL_PREFIX=tokenizers/sp_mod13k
TAR_SP_MODEL_PREFIX=tokenizers/sp_orig13k
SHARDS=5
LOWERCASE=True
TFRECORDS_DIRECTORY=data_clean/tfrecords

echo "==== Constructing data for training use ===="

python3 build_data.py \
	--input_text=${INPUT_TEXT} \
	--target_text=${TARGET_TEXT} \
	--train_inp_tokenizer=${TRAIN_INP_TOKENIZER} \
	--train_tar_tokenizer=${TRAIN_TAR_TOKENIZER} \
	--input_vocab_size=${INPUT_VOCAB_SIZE} \
	--target_vocab_size=${TARGET_VOCAB_SIZE} \
	--inp_sp_model_prefix=${INP_SP_MODEL_PREFIX} \
	--tar_sp_model_prefix=${TAR_SP_MODEL_PREFIX} \
	--shards=${SHARDS} \
	--lowercase=${LOWERCASE} \
	--tfrecords_directory=${TFRECORDS_DIRECTORY}

echo " ==== Happy training! ===="
