#!/bin/sh

echo "=== Setting up configuration ==="

# Saving the model stuff.
# We actually default to datetime to prevent model overwrite.
# MODEL_NAME=dmod512_dff1024_4lyr_4hd

# Data stuff
INPUT_TEXT='data_clean/modern.txt'
TARGET_TEXT='data_clean/original.txt'
INP_SP_MODEL_FILE=tokenizers/modern2k.model
TAR_SP_MODEL_FILE=tokenizers/original2k.model
INP_NBEST_SIZE=5
TAR_NBEST_SIZE=5

# Model stuff
D_MODEL=512
NUM_HEADS=4
D_FFN=1024
NUM_LAYERS=4
DROPOUT_RATE=0.1

# Learning parameter stuff
WARMUP_STEPS=4000
EPOCHS=21
MAXLEN=100

echo "=== Beginning training ==="
python3 train.py \
  --input_text=${INPUT_TEXT} \
  --target_text=${TARGET_TEXT} \
  --inp_nbest_size=${INP_NBEST_SIZE} \
  --tar_nbest_size=${TAR_NBEST_SIZE} \
  --inp_sp_model_file=${INP_SP_MODEL_FILE} \
  --tar_sp_model_file=${TAR_SP_MODEL_FILE} \
  --d_model=${D_MODEL} \
  --num_heads=${NUM_HEADS} \
  --d_ffn=${D_FFN} \
  --num_layers=${NUM_LAYERS} \
  --dropout_rate=${DROPOUT_RATE} \
  --warmup_steps=${WARMUP_STEPS} \
  --opt_name=${OPT_NAME} \
  --maxlen=${MAXLEN} \
  # --model_name=${MODEL_NAME} \

echo "=== Finished training. Congrats! ==="
