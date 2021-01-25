import logging
logging.info("\n\n~~~~~~~~ Importing Modules ~~~~~~~~\n")

import os
import json
import time
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
from data_utils import DataManager
from utils import print_bar
from model import Transformer
from utils import create_look_ahead_mask, create_padding_mask

from absl import flags
from absl import app

FLAGS = flags.FLAGS

### Checkpointing and TensorBoarding parameter flag(s)
flags.DEFINE_string('model_name', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    help='Model name for saving to checkpoints and log files. Defaults to current time.')

### Data path locaation flags
flags.DEFINE_string('tfrecords_directory', default='', help='Path of training dataset tfrecords directory')
flags.DEFINE_string('inp_sp_model_prefix', default='', help='Input sentencepiece model file prefix.')
flags.DEFINE_string('tar_sp_model_prefix', default='', help='Target sentencepiece model file prefix.')

### Transformer model parameter flags
flags.DEFINE_integer('d_model', default=512, help='Embedding dimension. Used in attention layers.')
flags.DEFINE_integer('num_heads', default=8, help='Number of heads to use in MultiHeadAttention.')
flags.DEFINE_integer('d_ffn', default=2048, help='Dimension of pointwise feed forward networks.', lower_bound=1)
flags.DEFINE_integer('num_layers', default=6, help='Number of stochastic blocks/encoder layers.', lower_bound=0)
flags.DEFINE_float('dropout_rate', default=0.1, help='Rate to drop units.')

### Adaptive softmax specific parameters
flags.DEFINE_multi_integer('cutoffs', default=[],
help='Cutoffs to use for adaptive softmax layer. Do NOT\
         enter the final cutoff (the vocab size). This will \
         be inferred from your sp_model_file. Cutoffs may be \
         entered by repated use of --cutoffs=[NUMBER].')
flags.DEFINE_integer('proj_factor', default=4, help='Reduction factor of d_model in adaptive softmax for successive clusters')
flags.DEFINE_multi_integer('proj_dims', default=[], help='Manually set reduction factors. Must match number of clusters.')
flags.DEFINE_integer('max_position', default=512, help='Max size of input or target sequence ever used.')

### Learning parameters
# flags.DEFINE_float('max_lr', default=1e-4, help='Maximum learning rate after warmup. Used in CosineSchedule.')
flags.DEFINE_integer('warmup_steps', default=4000, help='Number of warmup steps for the learning rate.')
flags.DEFINE_integer('epochs', default=20, help='Number of epochs')
flags.DEFINE_string('opt_name', default='adam', help='Available choices are set by the tf.keras.optimizers.get() call.')
flags.DEFINE_integer('batch_size', default=32, help='Batch size.')
flags.DEFINE_integer('maxlen', default=50, help='Maximum input or target sequence length. \
For computing and analyzing data length statistics, please see the methods of the DataManager class')

# S3
# DEEP LEARNING AMI
# p2 xlarge is the smallest machine learning
# have to be in the .ssh folder when ssh'ing. use ssh -i enter_thing_here

### Custom learning rate schedulers
class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "warmup_steps": self.warmup_steps}
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, decay_steps, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.pi = 3.1415927
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "warmup_steps": self.warmup_steps}
    def __call__(self, step):
        linear = self.max_lr*(step/self.warmup_steps)
        angle = self.pi*tf.math.maximum(step-self.warmup_steps, 0)/self.decay_steps
        cosine = 0.5*self.max_lr*(1+tf.math.cos(angle))
        return tf.math.minimum(linear, cosine)

def loss_fn(labels, logits):
    mask = tf.not_equal(labels, 0)
    loss = loss_fn.scc(labels, logits)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)
loss_fn.scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def accuracy_fn(labels, logits):
    mask = tf.not_equal(labels, 0)
    preds = tf.cast(tf.argmax(logits, axis=-1), labels.dtype)
    acc = tf.cast(tf.equal(labels, preds), tf.float32)
    acc = tf.boolean_mask(acc, mask)
    return tf.reduce_mean(acc)

def main(argv):
    if FLAGS.d_model%FLAGS.num_heads:
        raise ValueError('Number of heads must divide d_model')

    data_config = {
        'tfrecords_directory':FLAGS.tfrecords_directory,
        'inp_sp_model_prefix':FLAGS.inp_sp_model_prefix,
        'tar_sp_model_prefix':FLAGS.tar_sp_model_prefix
    }
    dm = DataManager.initialize_from_tfrecord(data_config)
    input_vocab_size = int(dm.inp_tokenizer.vocab_size())
    target_vocab_size = int(dm.tar_tokenizer.vocab_size())

    # Take care of additional constraints on inputs that needed the vocab size
    if any([z>=target_vocab_size for z in FLAGS.cutoffs]) or len(set(FLAGS.cutoffs))!=len(FLAGS.cutoffs):
        raise ValueError(f"Cutoffs must not exceed {target_vocab_size} or contain duplicates.")
    if FLAGS.cutoffs:
        FLAGS.cutoffs.sort() # this is redundant, the layer sorts anyway. but to be safe...
        FLAGS.cutoffs.append(target_vocab_size)

    ### Define learning rate schedule and simulated annealing schedule for gumbel softmax temperature tau.
    logging.info(f"\n\nInitializing {FLAGS.opt_name} optimizer with {FLAGS.warmup_steps} warmup steps.")
    learning_rate = TransformerSchedule(FLAGS.d_model, FLAGS.warmup_steps)
    optimizer = tf.keras.optimizers.get(FLAGS.opt_name)
    optimizer.learning_rate = learning_rate
    optimizer.clipvalue = 0.1

    # Setup the model
    tf.keras.backend.clear_session()
    config = {
        'num_layers': FLAGS.num_layers,
        'd_model':FLAGS.d_model,
        'num_heads': FLAGS.num_heads,
        'd_ffn': FLAGS.d_ffn,
        'input_vocab_size': input_vocab_size,
        'target_vocab_size': target_vocab_size,
        'pe_input': FLAGS.max_position,
        'pe_target': FLAGS.max_position,
        'p_drop': FLAGS.dropout_rate,
        # 'cutoffs': FLAGS.cutoffs,
        # 'proj_factor': FLAGS.proj_factor,
        # 'proj_dims': FLAGS.proj_dims,
    }
    logging.info("\n\nInitializing model...")
    logging.info(f"Model parameters:\n{config}")
    model = Transformer(**config)

    # Define metrics
    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.Mean()
    valid_acc = tf.keras.metrics.Mean()

    logging.info("\n\nDefining training and evaluation steps...")
    LOOKAHEAD_MASK = create_look_ahead_mask(FLAGS.max_position)
    @tf.function
    def train_step(inp, tar):
        tar_in = tar[:, :-1]
        tar_out = tar[:, 1:]
        tar_len = tf.shape(tar_in)[1]
        lookahead = LOOKAHEAD_MASK[:, :, :tar_len, :tar_len]
        padding = create_padding_mask(inp)
        with tf.GradientTape() as tape:
            logits, _ = model(inp, tar_in, training=True, enc_padding_mask=padding,
                           look_ahead_mask=lookahead, dec_padding_mask=padding)
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
        lookahead = LOOKAHEAD_MASK[:, :, :tar_len, :tar_len]
        padding = create_padding_mask(inp)
        logits, _ = model(inp, tar_in, training=False, enc_padding_mask=padding,
                       look_ahead_mask=lookahead, dec_padding_mask=padding)
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

    # # TODO: FIGURE OUT WHAT TO DO HERE, HOW TO LOAD TensorBoard during a script
    # If in colab write:
    # %load_ext tensorboard
    # %tensorboard --logdir logs/

    # Configure datasets for training
    logging.info("\n\nConfiguring datasets for training...")
    logging.info(f"\nSetting max input/target sequence length to: {FLAGS.maxlen}")
    glob_step = tf.Variable(0, dtype=tf.int64) # This will break tf.summary if we use int32
    train_ds, valid_ds = dm.get_train_valid_datasets(FLAGS.maxlen)

    # print out a model summary now that we know what the input data looks like
    temp_inp, temp_tar = next(iter(train_ds.batch(1)))
    model(temp_inp, temp_tar[:, :-1], False, None, None, None)
    logging.info(model.summary())

    train_ds = train_ds.cache().shuffle(10000).batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    DATASET_SIZE = None
    iterator=iter(train_ds)

    # Set up checkpointing to periodically save the model every epoch
    checkpoint_path = "./checkpoints/train/"+FLAGS.model_name
    logging.info(f"\n\nInitializing checkpoints. Models will be saved to {checkpoint_path}")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, glob_step=glob_step, iterator=iterator)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        try:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            logging.info('Latest checkpoint restored!!')
        except:
            logging.warning("Model may have changed, could not restore checkpoint.")

    ckpt_save_path = ckpt_manager.save()
    logging.info(f'Checkpointing model initialization at {ckpt_save_path}')
    with open(checkpoint_path+'/config', 'w') as file:
        file.write(json.dumps(config))
    logging.info(f"Writing model configuration to {checkpoint_path+'/config'}")

    # Run the actual training loop!
    absolute_start = time.time()
    logging.info("\n\n~~~~~~~~~~ Beginning training ~~~~~~~~~~")
    for epoch in range(FLAGS.epochs):

        logging.info('\n'+'-'*10+f' Epoch {epoch+1}/{FLAGS.epochs} '+'-'*10)
        start = time.time()
        for metric in [train_loss, valid_loss, train_acc, valid_acc]:
            metric.reset_states()

        for step, (inp, tar) in enumerate(train_ds):
            train_step(inp, tar)
            diff = (time.time()-start)/(step+1)
            print_bar(step, DATASET_SIZE, diff, train_loss.result().numpy())

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=glob_step)
                tf.summary.scalar('accuracy', train_acc.result(), step=glob_step)
                tf.summary.scalar('lr', learning_rate(tf.cast(glob_step, tf.float32)), step=glob_step)
            glob_step.assign_add(1)

            # if (step+1)%1000==0:
            #     try:
            #         os.mkdir('plots')
            #     except:
            #         pass
            #     logging.info(f"Global step:, {int(glob_step)}. Saving plots...")
            #     visualize_pi_weights(model)
            #     plt.savefig(f"plots/step{int(glob_step)}.png")

        if DATASET_SIZE is None:
            DATASET_SIZE = int(glob_step)

        for inp, tar in valid_ds:
            evaluation_step(inp, tar)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=glob_step)
            tf.summary.scalar('accuracy', valid_acc.result(), step=glob_step)

        ckpt_save_path = ckpt_manager.save()
        logging.info(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    tot_time = time.time()-absolute_start
    minutes = int(tot_time)//60
    seconds = int(tot_time)%60
    logging.info('*'*100+"\n\nTRAINING COMPLETE.\n\n"+'*'*100)
    try:
        os.mkdir('saved_models')
    except:
        pass
    logging.info(f"Saving final model to {'saved_models/'+FLAGS.model_name}")
    model.save('saved_models/'+FLAGS.model_name)
    logging.info(f"\n\nTotal time: {minutes:.02d}min. {seconds:.02d}sec.\n\n")

if __name__=="__main__":
    app.run(main)
