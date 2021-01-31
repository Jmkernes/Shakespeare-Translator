import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("_supplemental_training.log"),
        logging.StreamHandler()
    ]
)

logging.info("\n\n~~~~~~~~ Importing Modules ~~~~~~~~\n")

import os
import json
import time
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
from data_utils import DataManager, load_sentencepiece_model
from utils import print_bar, predict, create_masks
from model import Transformer

from absl import flags
from absl import app

FLAGS = flags.FLAGS

### Checkpointing and TensorBoarding parameter flag(s)
flags.DEFINE_string('model_name', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    help='Model name for saving to checkpoints and log files. Defaults to current time.')

### Data path location flags
flags.DEFINE_string('input_text', default='', help='Location of input text file.')
flags.DEFINE_string('target_text', default='', help='Location of target text file.')
flags.DEFINE_string('inp_sp_model_file', default='', help='Input sentencepiece model file prefix.')
flags.DEFINE_string('tar_sp_model_file', default='', help='Target sentencepiece model file prefix.')

### Tokenizer regularization
flags.DEFINE_integer('inp_nbest_size', default=5, help='SentencePiece tokenization sampler for inputs')
flags.DEFINE_integer('tar_nbest_size', default=5, help='SentencePiece tokenization sampler for targets')

### Transformer model parameter flags
flags.DEFINE_integer('d_model', default=512, help='Embedding dimension. Used in attention layers.')
flags.DEFINE_integer('num_heads', default=4, help='Number of heads to use in MultiHeadAttention.')
flags.DEFINE_integer('d_ffn', default=1024, help='Dimension of pointwise feed forward networks.', lower_bound=1)
flags.DEFINE_integer('num_layers', default=4, help='Number of stochastic blocks/encoder layers.', lower_bound=0)
flags.DEFINE_float('dropout_rate', default=0.1, help='Rate to drop units.')
flags.DEFINE_integer('max_position', default=512, help='Max size of input or target sequence model might ever use.')

### Learning parameters
flags.DEFINE_integer('warmup_steps', default=1000, help='Number of warmup steps for the learning rate.')
flags.DEFINE_integer('epochs', default=21, help='Number of epochs')
flags.DEFINE_integer('batch_size', default=32, help='Batch size.')
flags.DEFINE_integer('maxlen', default=100, help='Maximum sequence length for both inputs and targets.')
flags.DEFINE_float('val_frac', default=0.04, help='Percent of data to reserve for validation. A float in [0,1).')

def loss_fn(labels, logits):
    mask = tf.not_equal(labels, 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def accuracy_fn(labels, logits):
    mask = tf.not_equal(labels, 0)
    preds = tf.cast(tf.argmax(logits, axis=-1), labels.dtype)
    acc = tf.cast(tf.equal(labels, preds), tf.float32)
    acc = tf.boolean_mask(acc, mask)
    return tf.reduce_mean(acc)

class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, **kwargs):
        super().__init__(**kwargs)
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "warmup_steps": self.warmup_steps}
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)/ 10.

def main(argv):
    if FLAGS.d_model%FLAGS.num_heads:
        raise ValueError('Number of heads must divide d_model')

    # Initialize tokenizers and dataset
    data_config = {
        'input_text':FLAGS.input_text,
        'target_text':FLAGS.target_text,
        'inp_sp_model_file':FLAGS.inp_sp_model_file,
        'tar_sp_model_file':FLAGS.tar_sp_model_file,
        'inp_nbest_size':FLAGS.inp_nbest_size,
        'tar_nbest_size':FLAGS.tar_nbest_size
    }
    dm = DataManager.directly_from_text(data_config)
    input_vocab_size = int(dm.inp_tokenizer.vocab_size())
    target_vocab_size = int(dm.tar_tokenizer.vocab_size())

    # Initialize the optimizer
    warmup_steps = FLAGS.warmup_steps
    learning_rate = TransformerSchedule(d_model=d_model, warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Initialize the model
    tf.keras.backend.clear_session()
    config = {
        'num_layers': FLAGS.num_layers,
        'd_model': FLAGS.d_model,
        'num_heads': FLAGS.num_heads,
        'd_ffn': FLAGS.d_ffn,
        'input_vocab_size': input_vocab_size,
        'target_vocab_size': target_vocab_size,
        'pe_input': FLAGS.max_position,
        'pe_target': FLAGS.max_position,
        'p_drop': FLAGS.dropout_rate
    }
    logging.info("\n\nInitializing model...")
    logging.info(f"Model parameters:\n{config}")
    model = Transformer(**config)

    # Define metrics
    logging.info("\n\nDefining training and evaluation steps, and initializing metrics...")
    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.Mean()
    valid_acc = tf.keras.metrics.Mean()

    @tf.function
    def train_step(inp, tar):
        tar_in = tar[:, :-1]
        tar_out = tar[:, 1:]
        tar_len = tf.shape(tar_in)[1]
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar_in)
        with tf.GradientTape() as tape:
            logits, _ = model(inp, tar_in, training=True, enc_padding_mask=enc_padding_mask,
                           look_ahead_mask=look_ahead_mask, dec_padding_mask=dec_padding_mask)
            loss = loss_fn(tar_out, logits)
        accuracy = accuracy_fn(tar_out, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(accuracy)

    @tf.function
    def evaluation_step(inp, tar):
        tar_in = tar[:, :-1]
        tar_out = tar[:, 1:]
        tar_len = tf.shape(tar_in)[1]
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar_in)
        logits, _ = model(inp, tar_in, training=False, enc_padding_mask=enc_padding_mask,
                       look_ahead_mask=look_ahead_mask, dec_padding_mask=dec_padding_mask)
        loss = loss_fn(tar_out, logits)
        accuracy = accuracy_fn(tar_out, logits)
        valid_loss(loss)
        valid_acc(accuracy)

    # Set up TensorBoard
    logging.info("\n\nInitializing TensorBoard...")
    train_log_dir = './logs/' + FLAGS.model_name + '/train'
    test_log_dir = './logs/' + FLAGS.model_name + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Configure datasets for training
    logging.info("\n\nConfiguring datasets for training...")
    logging.info(f"\nSetting max input/target sequence length to: {FLAGS.maxlen}")
    glob_step = tf.Variable(0, dtype=tf.int64) # This will break tf.summary if we use int32
    train_ds, valid_ds = dm.get_train_valid_datasets(FLAGS.maxlen, val_frac=FLAGS.val_frac)

    # Print out a model summary now that we know what the input data looks like
    temp_inp, temp_tar = next(iter(train_ds.batch(1)))
    model(temp_inp, temp_tar[:, :-1], False, None, None, None)
    logging.info(model.summary())

    # Finish configuring the datasets.
    DATASET_SIZE = int(dm.ds_size//batch_size)
    logging.info(f"\n\nShuffling, batching and caching data. Batch size: {FLAGS.batch_size}. Dataset size: {DATASET_SIZE} batches.")
    train_ds = train_ds.cache().shuffle(10000).batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    iterator=iter(train_ds)

    # Set up checkpointing to periodically save the model every epoch
    # We will implement a "keep only those that improve validation loss" later.
    checkpoint_path = "./checkpoints/train/"+FLAGS.model_name
    logging.info(f"\n\nInitializing checkpoints. Models will be saved to {checkpoint_path}")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, glob_step=glob_step, iterator=iterator)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        try:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            logging.info('Latest checkpoint restored!')
        except:
            logging.warning("Model may have changed, could not restore checkpoint.")
    ckpt_save_path = ckpt_manager.save()
    logging.info(f'Checkpointing model initialization at {ckpt_save_path}')

    # Save model configuration to json file
    with open(checkpoint_path+'/config.json', 'w') as file:
        file.write(json.dumps(config))
    logging.info(f"Writing model configuration to {checkpoint_path+'/config'}")

    ##################################
    # Run the actual training loop!  #
    ##################################
    best_val_loss = float('inf')
    absolute_start = time.time()
    print("\n\n~~~~~~~~~~ Beginning training ~~~~~~~~~~")
    for epoch in range(epochs):
        print('\n'+'-'*10+f' Epoch {epoch+1}/{epochs} '+'-'*10)

        for metric in [train_loss, valid_loss, train_acc, valid_acc]:
            metric.reset_states()

        start = time.time()
        for step, (inp, tar) in enumerate(train_ds):

            train_step(inp, tar)

            diff = (time.time()-start)/(step+1)
            print_bar(step, DATASET_SIZE, diff, train_loss.result().numpy())
            if (int(glob_step)+1)%100==0:
                step = int(glob_step)
                iter_message = f"Iteration {step+1:02d}/{DATASET_SIZE*epochs}:"
                time_message = f" {1/diff:.2f} it/s."
                loss_message = f" Loss: {float(train_loss.result()):.3f}"
                acc_message = f" Accuracy: {float(train_acc.result()):.3f}"
                print(iter_message+time_message+loss_message+acc_message)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=glob_step)
                tf.summary.scalar('accuracy', train_acc.result(), step=glob_step)
                tf.summary.scalar('lr', learning_rate(tf.cast(glob_step, tf.float32)), step=glob_step)
            glob_step.assign_add(1)

            if int(glob_step)%500==0:
                sentence = 'where are you?'
                pred_sentence = dm.tar_tokenizer.detokenize(predict(sentence, dm.inp_tokenizer, model)[0])
                print(f"Input sentence: {sentence}")
                print(f"Output sentence: {pred_sentence.numpy()[0].decode()}")

        for inp, tar in valid_ds:
            evaluation_step(inp, tar)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=glob_step)
            tf.summary.scalar('accuracy', valid_acc.result(), step=glob_step)

        curr_val_loss = float(valid_loss.result())
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            ckpt_save_path = ckpt_manager.save()
            logging.info(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        else:
            logging.info(f'Validation loss did not improve, skipping checkpoint.')

    tot_time = time.time()-absolute_start
    minutes = int(tot_time)//60
    seconds = int(tot_time)%60
    print('*'*100+"\n\nTRAINING COMPLETE.\n\n"+'*'*100)
    print(f"\n\nTotal time: {minutes:02d}min. {seconds:02d}sec.\n\n")

if __name__=="__main__":
    app.run(main)
