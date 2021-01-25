#!/bin/sh

echo "=== Setting up configuration ==="

# Data stuff
TFRECORDS_DIRECTORY=data_clean/tfrecords
INP_SP_MODEL_PREFIX=tokenizers/sp_mod16k
TAR_SP_MODEL_PREFIX=tokenizers/sp_orig16k

# Model stuff
D_MODEL=256
NUM_HEADS=4
D_FFN=2048
NUM_LAYERS=4
DROPOUT_RATE=0.1
# CUTOFF1=250
# CUTOFF2=2500
# PROJ_FACTOR=4

# Don't set proj_dims
WARMUP_STEPS=4000
EPOCHS=10
OPT_NAME=adam
MAXLEN=75

# File prefix for checkpointing and TensorBoard
MODEL_NAME=dmodel256_dffn2048_blocks12

echo "=== Beginning training ==="
python3 train.py \
  --tfrecords_directory=${TFRECORDS_DIRECTORY} \
  --inp_sp_model_prefix=${INP_SP_MODEL_PREFIX} \
  --tar_sp_model_prefix=${TAR_SP_MODEL_PREFIX} \
  --d_model=${D_MODEL} \
  --num_heads=${NUM_HEADS} \
  --d_ffn=${D_FFN} \
  --num_layers=${NUM_LAYERS} \
  --dropout_rate=${DROPOUT_RATE} \
  --warmup_steps=${WARMUP_STEPS} \
  --opt_name=${OPT_NAME} \
  --epochs=${EPOCHS} \
  --maxlen=${MAXLEN} \
  # --cutoffs=${CUTOFF1} \
  # --cutoffs=${CUTOFF2} \
  # --proj_factor=${PROJ_FACTOR} \
  # --model_name=${MODEL_NAME} \

echo "=== Finished training. Congrats! ==="
