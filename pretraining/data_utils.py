import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

def load_sentencepiece_model(model_proto, nbest_size):
    proto = tf.io.gfile.GFile(model_proto, 'rb').read()
    return tf_text.SentencepieceTokenizer(model=proto, nbest_size=nbest_size)

def add_sos_eos(line):
        """ WARNING: This assume that sos and eos are 1 and 2 respectively. """
        dtype = line.dtype
        sos = tf.constant([1], dtype=dtype)
        eos = tf.constant([2], dtype=dtype)
        return tf.concat([sos, line, eos], axis=0)

def add_padding(line, maxlen):
    n = tf.maximum(maxlen-tf.size(line), 0)
    return tf.pad(line, [(0, n)])

def create_cloze_masks(inp, p_select=0.15, p_mask=0.8, p_keep=0.1, p_rand=0.1):
    """ p_mask+p_keep+p_rand=1 must be true. p_select is prob to mess with a token """
    bound1 = p_select*p_mask
    bound2 = bound1 + p_select*p_keep
    bound3 = bound2 + p_select*p_rand
    temp_mask = tf.random.uniform(tf.shape(inp))
    mask_mask = temp_mask<bound1
    keep_mask = tf.math.logical_and(temp_mask<bound2, temp_mask>=bound1)
    rand_mask = tf.math.logical_and(temp_mask<bound3, temp_mask>=bound2)
    return mask_mask, keep_mask, rand_mask

def create_cloze_data(inp, mask_id, vocab_size, **kwargs):
    mask_mask, keep_mask, rand_mask = create_cloze_masks(inp, **kwargs)
    random_ids = tf.random.uniform(tf.shape(inp), 0, vocab_size, inp.dtype)
    mask_ids = tf.fill(tf.shape(inp), mask_id)
    target_mask = tf.logical_or(tf.logical_or(mask_mask, keep_mask), rand_mask)
    target_mask = tf.logical_and(target_mask, tf.not_equal(inp, 0))
    tar = inp # the goal is to reconstruct original input
    inp = tf.where(mask_mask, mask_ids, inp)
    inp = tf.where(rand_mask, random_ids, inp)
    return inp, tar, target_mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def load_dataset(text_file, sp, maxlen):
    vocab_size = int(sp.vocab_size())
    mask_id = vocab_size

    ds = tf.data.TextLineDataset(text_file)
    print('Tokenizing...')
    ds = ds.map(sp.tokenize)
    print(f'Filtering and padding to max length: {maxlen}...')
    ds = ds.map(add_sos_eos).filter(lambda x: tf.size(x) <= maxlen)
    ds = ds.map(lambda x: add_padding(x, maxlen))
    print(f'Generating cloze data...')
    ds = ds.map(lambda x: create_cloze_data(x, mask_id, vocab_size))
    return ds

def find_dataset_size(ds):
    tot = 0
    for x, _, _ in ds.batch(4096):
        tot += x.shape[0]
    return tot

class DataManager:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.nbest_size = int(tokenizer.nbest_size)
        self.vocab_size = int(tokenizer.vocab_size())
        self.mask_id = self.vocab_size
        self.max_length = int(next(iter(dataset))[0].shape[0])
        self.ds_size = find_dataset_size(dataset)

    @classmethod
    def from_text(cls, text_file, sp_model_file, nbest_size, max_length):
        print('-'*5+' loading tokenizer '+'-'*5)
        tokenizer = load_sentencepiece_model(sp_model_file, nbest_size)
        print('-'*5+' loading data '+'-'*5)
        dataset = load_dataset(text_file, tokenizer, max_length)
        return cls(dataset, tokenizer)

    def get_batch_ds(self, batch_size, buffer_size=50000):
        return self.dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    def get_train_test_datasets(self, batch_size, buffer_size=50000, frac=0.01):
        batched_ds = self.get_batch_ds(batch_size, buffer_size)
        ds_size = self.ds_size//batch_size
        test_ds_size = int(frac*ds_size)
        test_ds = batched_ds.take(test_ds_size)
        train_ds = batched_ds.skip(test_ds_size)
        return train_ds, test_ds
