import os
import time
import logging
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

# Train a sentencepiece model
def build_sentencepiece_model(input_file, model_prefix, vocab_size, lowercase):
    if not os.path.isfile(input_file):
        raise FileExistsError("Input text file does not exist.")
    if not input_file.endswith('.txt'):
        raise ValueError("Input text file must end with .txt extension.")

    import sentencepiece as spm

    if model_prefix is None:
        model_prefix = f"wiki2_{vocab_size}"
    normalization_rule = 'nfkc_cf' if lowercase else 'nfkc'

    logging.info("\n\nTraining SentencePiece on file: {input_file}...\n\n")
    start = time.time()
    spm.SentencePieceTrainer.Train(
        f'--input={input_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--user_defined_symbols=@-@ '
        f'--normalization_rule_name={normalization_rule} '
        f'--pad_id=0 --unk_id=3 --bos_id=1 --eos_id=2 '
        f'--pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] '
    )
    logging.info(f"""\n\nFinished training. Time: {time.time()-start:.2f}s.
     Model protobuf saved to: {model_prefix}.model.
     Vocab size: {vocab_size}.\n\n""")
    return f"{model_prefix}.model"


def load_sentencepiece_model(model_proto, nbest_size=1):
    if not os.path.isfile(model_proto):
        raise FileExistsError(f"The sentencepiece model file '{model_proto}' does not exist.")
    proto = tf.io.gfile.GFile(model_proto, 'rb').read()
    return tf_text.SentencepieceTokenizer(model=proto, nbest_size=nbest_size)

def find_dataset_size(dataset):
    return tf.cast(dataset.reduce(0, lambda x, _: x+1), tf.int64)

def get_dataset_type(dataset):
    try:
        return next(iter(dataset)).dtype
    except:
        raise ValueError("Dataset is empty.")

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_line(line):
    inp, tar = line
    inp = tf.io.serialize_tensor(inp)
    tar = tf.io.serialize_tensor(tar)
    feature = {
        'modern': _bytes_feature(inp),
        'original': _bytes_feature(tar)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def deserialize_line(example_proto, out_type=None):
    """ If using sentencepiece, convert to int32, if using TextVectorizationLayer use int64 """
    out_type = tf.int32 if out_type is None else out_type
    feature_description = {
        'modern': tf.io.FixedLenFeature([], tf.string, default_value=b''),
        'original': tf.io.FixedLenFeature([], tf.string, default_value=b'')
    }
    proto = tf.io.parse_single_example(example_proto, feature_description)
    modern = tf.io.parse_tensor(proto['modern'], out_type=out_type)
    original = tf.io.parse_tensor(proto['original'], out_type=out_type) # change to 64 for vectorizeLayer
    return modern, original

def write_to_tfrecord(dataset, filename, dataset_size=None):
    if dataset_size is None:
        dataset_size = find_dataset_size(dataset)
    with tf.io.TFRecordWriter(filename) as writer:
        start = time.time()
        for i, x in enumerate(dataset):
            proto = serialize_line(x)
            writer.write(proto)
            end = '\n' if i == dataset_size-1 else '\r'
            print(f"{100*(i+1)/dataset_size:.2f}% complete", end=end)
        logging.info(f"Writing time: {time.time()-start:.2f}s\n")

def write_to_tfrecord_shards(directory, dataset, shards):
    """ Saves data into shards of equal length. Will truncate so data loss could be up to shards
    number of lines (not too big a deal for shards O(1) and ds_size O(1e4))"""

    logging.info(f"\n\nCreating tfrecords directory: {directory}.")
    try:
        os.mkdir(directory)
    except:
        pass

    ds_size = find_dataset_size(dataset)
    shard_size = ds_size//shards

    start = time.time()
    for i in range(shards):
        filename = os.path.join(directory, f'file{i+1}.tfrecords')
        shard_ds = dataset.skip(i*shard_size).take(shard_size)

        logging.info(f"\n\n--- Writing shard {i+1} to {filename} ---")
        write_to_tfrecord(shard_ds, filename, shard_size)

    logging.info(f"Total time to convert data to tfrecords: {time.time()-start:.2f}s\n\n")

def read_from_tfrecord_directory(directory, out_type=None):
    if not os.path.isdir(directory):
        raise FileExistsError("The specifed path does not point to a directory.")
    filenames = os.listdir(directory)
    filenames = [os.path.join(directory, name) for name in filenames]
    file_ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = file_ds.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: deserialize_line(x, out_type=out_type))
    return ds

def get_inp_tar_text_ds(input_text, target_text):
    if not os.path.isfile(input_text) or not os.path.isfile(target_text):
        raise FileExistsError('One of the specified text files does not exist.')
    if not input_text.endswith('.txt') or not target_text.endswith('.txt'):
        raise ValueError("Text files must end with a .txt extension.")
    inp_ds = tf.data.TextLineDataset(input_text)
    tar_ds = tf.data.TextLineDataset(target_text)
    return tf.data.Dataset.zip((inp_ds, tar_ds))

class DataManager:
    def __init__(self, dataset, inp_tokenizer, tar_tokenizer):
        self.dataset = dataset
        self.inp_tokenizer = inp_tokenizer
        self.tar_tokenizer = tar_tokenizer
        self.ds_size = find_dataset_size(dataset)
        self.in_lens = None
        self.tar_lens = None

    @classmethod
    def initialize_from_text(cls, configs):
        input_text = configs['input_text']
        target_text = configs['target_text']
        train_inp_tokenizer = configs['train_inp_tokenizer']
        train_tar_tokenizer = configs['train_tar_tokenizer']
        input_vocab_size = configs['input_vocab_size']
        target_vocab_size = configs['target_vocab_size']
        inp_sp_model_prefix = configs['inp_sp_model_prefix']
        tar_sp_model_prefix = configs['tar_sp_model_prefix']
        shards = configs['shards']
        lowercase = configs['lowercase']
        tfrecords_directory = configs['tfrecords_directory']

        if not configs['train_inp_tokenizer']:
            inp_sp_model_file = inp_sp_model_prefix+'.model'
        else:
             inp_sp_model_file = build_sentencepiece_model(
                input_text, inp_sp_model_prefix, input_vocab_size, lowercase
            )

        if not configs['train_tar_tokenizer']:
            tar_sp_model_file = tar_sp_model_prefix+'.model'
        else:
             tar_sp_model_file = build_sentencepiece_model(
                target_text, tar_sp_model_prefix, target_vocab_size, lowercase
            )

        text_ds = get_inp_tar_text_ds(input_text, target_text)
        inp_tokenizer = load_sentencepiece_model(inp_sp_model_file)
        tar_tokenizer = load_sentencepiece_model(tar_sp_model_file)
        tokenized_ds = text_ds.map(
            lambda x,y: (inp_tokenizer.tokenize(x), tar_tokenizer.tokenize(y)))
        write_to_tfrecord_shards(tfrecords_directory, tokenized_ds, shards)

        tfrecord_configs = {
            'tfrecords_directory': tfrecords_directory,
            'inp_sp_model_prefix': inp_sp_model_prefix,
            'tar_sp_model_prefix': tar_sp_model_prefix
        }
        return cls.initialize_from_tfrecord(tfrecord_configs)

    @classmethod
    def directly_from_text(cls, configs):
        input_text = configs['input_text']
        target_text = configs['target_text']
        inp_sp_model_file = configs['inp_sp_model_file']
        tar_sp_model_file = configs['tar_sp_model_file']
        inp_nbest_size = configs['inp_nbest_size']
        tar_nbest_size = configs['tar_nbest_size']

        inp_tokenizer = load_sentencepiece_model(inp_sp_model_file, inp_nbest_size)
        tar_tokenizer = load_sentencepiece_model(tar_sp_model_file, tar_nbest_size)
        inp_ds = tf.data.TextLineDataset(input_text)
        tar_ds = tf.data.TextLineDataset(target_text)
        ds = tf.data.Dataset.zip((inp_ds, tar_ds))
        ds = ds.map(lambda x,y: (inp_tokenizer.tokenize(x), tar_tokenizer.tokenize(y)))
        return cls(ds, inp_tokenizer, tar_tokenizer)

    @classmethod
    def initialize_from_tfrecord(cls, configs):
        tfrecords_directory = configs['tfrecords_directory']
        inp_sp_model_file = configs['inp_sp_model_prefix']+'.model'
        tar_sp_model_file = configs['tar_sp_model_prefix']+'.model'

        logging.info(f"Loading tokenizers...")
        inp_tokenizer = load_sentencepiece_model(inp_sp_model_file)
        tar_tokenizer = load_sentencepiece_model(tar_sp_model_file)

        logging.info("Loading tfrecords from directory...")
        dataset = read_from_tfrecord_directory(configs['tfrecords_directory'])

        new_configs = {
            'dataset': dataset,
            'inp_tokenizer': inp_tokenizer,
            'tar_tokenizer': tar_tokenizer
        }
        return cls(**new_configs)

    def find_dataset_size(self, ds):
        return find_dataset_size(ds)

    def _compute_dataset_statistics(self):
        print("--- Computing sequence length statistics ---")
        in_lens, tar_lens = [], []
        for i, (x, y) in enumerate(self.dataset):
            in_lens.append(x.shape[0])
            tar_lens.append(y.shape[0])
            print(f"{100*(i+1)/self.ds_size:.2f}% complete", end='\r')
        self.in_lens = in_lens
        self.tar_lens = tar_lens

    def get_dataset_statistics(self):
        if self.in_lens is None:
            self._compute_dataset_statistics()
        in_mean, in_std = np.mean(self.in_lens), np.std(self.in_lens)
        out_mean, out_std = np.mean(self.tar_lens), np.std(self.tar_lens)
        print(f"\nInput sequences:\nMean: {in_mean:.2f}.\nStandard Deviation: {in_std:.2f}")
        print(f"\nTarget sequences:\nMean: {out_mean:.2f}.\nStandard Deviation: {out_std:.2f}")

    def size_with_maxlen(self, maxlen):
        if self.in_lens is None:
            self._compute_dataset_statistics()
        res = 0
        for i, (x, y) in enumerate(zip(self.in_lens, self.tar_lens)):
            if x <= maxlen and y <= maxlen:
                res += 1
        frac = np.round(res/self.ds_size, 2)
        print(f"New ds size: {res}. Covers {100*frac}% of original.")
        return res, frac

    @tf.function
    def _add_sos_eos(self, line):
        """ WARNING: This assume that sos and eos are 1 and 2 respectively. This is ok
        As these were defined in the above model. Note however, if you load your own
        SentencePiece model instead of using the methods above, you must make sure that
        You use the same convention."""
        dtype = line.dtype
        sos = tf.constant([1], dtype=dtype)
        eos = tf.constant([2], dtype=dtype)
        return tf.concat([sos, line, eos], axis=0)

    @tf.function
    def _add_padding(self, line, maxlen):
        n = tf.maximum(maxlen-tf.size(line), 0)
        return tf.pad(line, [(0, n)])

    def get_raw_dataset(self):
        return self.dataset

    def get_inp_tar_pairs(self, maxlen):
        ds = self.dataset.map(lambda x,y: (self._add_sos_eos(x), self._add_sos_eos(y)))
        ds = ds.filter(lambda x,y: tf.logical_and(tf.size(x)<=maxlen, tf.size(y)<=maxlen))
        ds = ds.map(lambda x,y: (self._add_padding(x, maxlen), self._add_padding(y, maxlen)))
        return ds

    def get_train_valid_datasets(self, maxlen, val_frac=0.01):
        """ Default is to retain 1% for validation set """
        ds = self.get_inp_tar_pairs(maxlen)
        ds_size = find_dataset_size(ds)
        val_size = int(float(ds_size)*val_frac)
        val_ds = ds.take(val_size)
        train_ds = ds.skip(val_size)
        return train_ds, val_ds
