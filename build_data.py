import logging
logging.info('-'*10, ' Importing modules ', '-'*10)
import os
import time
import tensorflow as tf
from data_utils import DataManager

from absl import flags
from absl import app

FLAGS = flags.FLAGS

# flags.DEFINE_boolean('use_defaults', False, 'Whether to use the default data build.')

# flags.DEFINE_boolean('download_data', False,
# 'Set to true to download wikitext2 from the web. Should only be run once.')

flags.DEFINE_boolean('train_inp_tokenizer', default=False, help='Train sentencepiece on input text.')
flags.DEFINE_boolean('train_tar_tokenizer', default=False, help='Train sentencepiece on target text.')
flags.DEFINE_string('input_text', default='', help='File path of input text.')
flags.DEFINE_string('target_text', default='', help='File path of target text.')
flags.DEFINE_integer('input_vocab_size', default=13000, help='Input vocab size. This flag is ignored if train_inp_tokenizer=False')
flags.DEFINE_integer('target_vocab_size', default=13000, help='Target vocab size. This flag is ignored if train_tar_tokenizer=False')
flags.DEFINE_string('inp_sp_model_prefix', default='', help='Location of sentencepiece model protobuff for inputs.')
flags.DEFINE_string('tar_sp_model_prefix', default='', help='Location of sentencepiece model protobuff for targets.')
flags.DEFINE_integer('shards', default=5, help='Number of shards to split tfrecords into.')
flags.DEFINE_boolean('lowercase', default=True, help='Whether to lowercase the input.')
flags.DEFINE_string('tfrecords_directory', default='', help='Name of directory to hold tfrecords files.')

def build(configs):
    logging.info(f"\n\n**** Creating dataset ****")
    logging.info(f"Input file: {configs['input_text']}")
    logging.info(f"Output file: {configs['target_text']}")

    start = time.time()
    dm = DataManager.initialize_from_text(configs)
    tot_time = time.time()-start

    logging.info("\n\n---- Sample from the dataset  ---")
    logging.info(next(iter(dm.dataset)))
    logging.info(f"\n\nTotal time to process data: {tot_time:.2f}s")
    return dm

def main(argv):
    if os.path.isdir(FLAGS.tfrecords_directory):
        logging.warning("The specified tfrecords directory already exists.")
        x = input("To proceed and overwrite, enter [y]. To exit enter any key:\n")
        if x!='y':
            quit()
    configs = {
        'input_text' : FLAGS.input_text,
        'target_text' : FLAGS.target_text,
        'train_inp_tokenizer' : FLAGS.train_inp_tokenizer,
        'train_tar_tokenizer' : FLAGS.train_tar_tokenizer,
        'input_vocab_size'  : FLAGS.input_vocab_size,
        'target_vocab_size' : FLAGS.target_vocab_size,
        'inp_sp_model_prefix': FLAGS.inp_sp_model_prefix,
        'tar_sp_model_prefix' : FLAGS.tar_sp_model_prefix,
        'shards' : FLAGS.shards,
        'lowercase' : FLAGS.lowercase,
        'tfrecords_directory' : FLAGS.tfrecords_directory
    }
    logging.info(f"\Current configurations: {configs}")
    build(configs)

if __name__=="__main__":
    app.run(main)
