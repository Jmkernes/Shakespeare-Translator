import time
import numpy as np
import tensorflow as tf
from utils import create_padding_mask, create_look_ahead_mask

def positionalEncoding(p_max, d_model, min_freq=1e-4):
    """ Positional encoding layer for the vanilla transformer
    Parameters:
        p_max (int): max sequence length (i.e. position)
        d_model (int): embedding dimension
        min_freq (default=1e-4): lowest frequency allowed.
            A larger number supports longer sequences
    Outputs:
        pos_encoding: a (p_max, d_model) tensor of embedding vectors
            for each integral position (row)
    """
    freqs = min_freq**(2.0*(np.arange(d_model)//2)/d_model)
    p = np.arange(p_max)
    matrix = p[:, np.newaxis]*freqs[np.newaxis, :]
    matrix[:,::2] = tf.math.sin(matrix[:,::2])
    matrix[:,1::2] = tf.math.cos(matrix[:,1::2])
    pos_encoding = tf.expand_dims(matrix, 0)
    pos_encoding = tf.cast(pos_encoding, tf.float32)
    return pos_encoding

def scaledDotProductAttention(Q, K, V, mask=None):
    """ Scaled dot product attention of the form
    ..math::
        \text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V
    Parameters:
        Q: (..., query_len, d_model) query tensor
        K: (..., key_len, d_model) key tensor
        V: (..., key_len, d_model) values tensor
    Outputs:
        Tensor of shape (..., query_len, d_model)
    """
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    logits = tf.matmul(Q, K, transpose_b=True)
    scaled_logits = logits/tf.math.sqrt(dk)
    if mask is not None:
        scaled_logits += -1e9*mask
    weights = tf.nn.softmax(scaled_logits, axis=-1)
    res = tf.matmul(weights, V)
    return res, weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multihead attention layer. Reshapes input to
    (batch_size, num_heads, seq_len, d_model) then applies
     scaled dot product attention
    Constructor parameters:
        d_model (int): embedding dimension
        num_heads (int): number of parallel heads. must divide d_model.
    Call parameters:
        Q: query tensor (..., query_len, d_model)
        K: key tensor (..., key_len, d_model)
        V: values tensor (..., key_len, d_values)
        mask: a mask of shape (1, 1, query_len, key_len), with 1's
            where attention should be zero and 0's elsewhere.
    Outputs:
        output: Tensor of shape (batch_size, query_len, d_model)
        attn_weights: Weights of shape (batch_size, num_heads, query_len, key_len)
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        assert d_model%num_heads==0, "Number of heads do not divide d_model"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model//num_heads

        self.Wq = tf.keras.layers.Dense(d_model)
        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split(self, A, batch_size):
        A = tf.reshape(A, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(A, [0,2,1,3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)

        Q = self.split(Q, batch_size)
        K = self.split(K, batch_size)
        V = self.split(V, batch_size)

        attn, weights = scaledDotProductAttention(Q, K, V, mask)
        attn = tf.transpose(attn, (0,2,1,3))
        attn = tf.reshape(attn, (batch_size, -1, self.d_model))

        output = self.dense(attn)

        return output, weights

def FeedForwardNetwork(d_model, d_ffn, activation='relu'):
    """ Fully connected layer. output feature dim= input feature dim
    Parameters:
        d_model: feature dimension
        d_ffn: hidden dimension of feed forward network
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(d_ffn, activation=activation),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    """ Single Encoder layer.
    Constructor parameters:
        d_model: embedding dimension
        num_heads: must divide d_model
        d_ffn: hidden dimension of feed forward network
        p_drop (default=0.1): dropout rate.
    Call parameters:
        x: input embedding (batch_size, seq_len, d_model)
        training (bool): true for training.
        mask: a (1, 1, query_len, key_len) boolean mask
    """
    def __init__(self, d_model, num_heads, d_ffn, p_drop=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(p_drop)
        self.dropout2 = tf.keras.layers.Dropout(p_drop)

    def call(self, x, training, mask):
        attn, _ = self.mha(x, x, x, mask)
        attn = self.dropout1(attn, training=training)
        out1 = self.layernorm1(x+attn)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1+ffn_output)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    """ Single Decoder layer.
    Constructor parameters:
        d_model: embedding dimension
        num_heads: must divide d_model
        d_ffn: hidden dimension of feed forward network
        p_drop (default=0.1): dropout rate.
    Call parameters:
        x: input embedding (batch_size, seq_len, d_model)
        enc_output: output from the encoder (batch_size, input_len, d_model)
        training (bool): true for training.
        pad_mask: a (1, 1, query_len, key_len) boolean mask to disregard
            padded inputs. Padded inputs must have ID of 0.
        look_ahead_mask: an upper triangular boolean mask used in first block
            of self-attention to keep queries q_{j<i} only for query q_i.
    """
    def __init__(self, d_model, num_heads, d_ffn, p_drop=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(p_drop)
        self.dropout2 = tf.keras.layers.Dropout(p_drop)
        self.dropout3 = tf.keras.layers.Dropout(p_drop)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, weights1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x+attn1)

        attn2, weights2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1+attn2)

        ffn = self.ffn(out2)
        ffn = self.dropout3(ffn, training=training)
        out3 = self.layernorm3(out2+ffn)

        return out3, weights1, weights2

class Encoder(tf.keras.layers.Layer):
    """ Stacked Encoders plus an embedding layer and positional pos_encoding
    Constructor parameters:
        num_layers: repeate EncodingLayer N times
        d_model: embedding dimension
        num_heads: must divide d_model
        d_ffn: hidden dimension of feed forward network
        input_vocab_size: max integer used in input vocabulary
        max_position: the maximum input sequence length
        p_drop (default=0.1): dropout rate.
    Call parameters:
        x: input embedding (batch_size, max_position, input_vocab_size)
        training (bool): true for training.
        mask: a (1, 1, query_len, key_len) boolean mask
    Outputs:
        Tensor of shape (batch_size, input_len, d_model)
    """
    def __init__(self, num_layers, d_model, num_heads, d_ffn,
                 input_vocab_size, max_position, p_drop=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positionalEncoding(max_position, d_model)
        self.embed = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.encoders = [EncoderLayer(d_model, num_heads, d_ffn, p_drop)
                         for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(p_drop)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embed(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:,:seq_len, :]

        x = self.dropout(x, training=training)

        for encode in self.encoders:
            x = encode(x, training, mask)

        return x

class Decoder(tf.keras.layers.Layer):
    """ Stacked Decoders plus an embedding layer and positional pos_encoding
    Constructor parameters:
        num_layers: repeate EncodingLayer N times
        d_model: embedding dimension
        num_heads: must divide d_model
        d_ffn: hidden dimension of feed forward network
        target_vocab_size: max integer used in target vocabulary
        max_position: the maximum input sequence length
        p_drop (default=0.1): dropout rate.
    Call parameters:
        x: input embedding (batch_size, max_position, target_vocab_size)
        training (bool): true for training.
        mask: a (1, 1, query_len, key_len) boolean mask
        pad_mask: a (1, 1, query_len, key_len) boolean mask to disregard
            padded inputs. Padded inputs must have ID of 0.
        look_ahead_mask: an upper triangular boolean mask used in first block
            of self-attention to keep queries q_{j<i} only for query q_i.
    Outputs:
        Tensor of shape (batch_size, target_len, d_model)
    """
    def __init__(self, num_layers, d_model, num_heads, d_ffn,
                 target_vocab_size, max_position, p_drop=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positionalEncoding(max_position, d_model)

        self.decoders = [DecoderLayer(d_model, num_heads, d_ffn, p_drop)
                             for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(p_drop)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        x = self.embed(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attn_weights = {}
        for i,decode in enumerate(self.decoders):
            x, block1, block2 = decode(x, enc_output, training,
                                       look_ahead_mask, padding_mask)
            attn_weights["decoder_layer{}_block1".format(i+1)] = block1
            attn_weights["decoder_layer{}_block2".format(i+1)] = block2

        return x, attn_weights #(batch_size, input_seq_len, d_model)

class Transformer(tf.keras.Model):
    """ Vanilla Transformer of stacked EncoderLayers followed by DecoderLayers.
    Constructor parameters:
        num_layers: repeat EncodingLayer N times
        d_model: embedding dimension
        num_heads: must divide d_model
        d_ffn: hidden dimension of feed forward network
        input_vocab_size: max integer used in input vocabulary
        target_vocab_size: max integer used in target vocabulary
        pe_input: the maximum positional encoding input length
        pe_target: the maximum positional encoding target length
        p_drop (default=0.1): dropout rate.
    Call parameters:
        inp: input sequence (batch_size, pe_input, input_vocab_size)
        tar: input sequence (batch_size, pe_target, target_vocab_size)
        training (bool): true for training.
        mask: a (1, 1, query_len, key_len) boolean mask
        enc_padding_mask: a (1, 1, query_len, key_len) boolean mask to disregard
            padded inputs. Padded inputs must have ID of 0.
        look_ahead_mask: a product of an upper triangular boolean mask used
            in first block of self-attention to keep queries q_{j<i} only for
            query q_i.look_ahead and a target padding mask.
        dec_padding_mask: A copy of the enc_padding mask for use in block2
            of the decoder
    Outputs:
        Tensor of shape (batch_size, target_len, d_model)
    """
    def __init__(self, num_layers, d_model, num_heads, d_ffn,
                 input_vocab_size, target_vocab_size, pe_input,
                 pe_target, p_drop=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ffn,
                              input_vocab_size, pe_input, p_drop)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ffn,
                              target_vocab_size, pe_target, p_drop)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask,
            dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attn_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attn_weights

class EncoderWithPretraining(tf.keras.layers.Layer):
    """ In contrast to layer used in the main model, this does not use an embedding layer """
    def __init__(self, num_layers, d_model, num_heads, d_ffn, max_position, p_drop, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positionalEncoding(max_position, d_model)
        self.encoders = [EncoderLayer(d_model, num_heads, d_ffn, p_drop) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(p_drop)

    def call(self, x, training, mask):
        # attn_weights = {}
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:,:seq_len, :]
        x = self.dropout(x, training=training)
        for i, encode in enumerate(self.encoders):
            x = encode(x, training, mask)
            # attn_weights[f'Encoder_layer_{i+1}'] = weights
        return x

class Embedding(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, vocab_size, max_position, p_drop, **kwargs):
        super().__init__(**kwargs)
        self.W_emb = self.add_weight(shape=(vocab_size, d_model), initializer='glorot_normal',
                                    trainable=True, name='Embeddings')
        self.encoder = EncoderWithPretraining(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ffn=d_ffn,
            max_position=max_position, p_drop=p_drop)

    def call(self, x, training, mask):
        x = tf.nn.embedding_lookup(self.W_emb, x)
        x = self.encoder(x, training, mask)
        return x

class TransformerWithPretraining(tf.keras.Model):
    """ The input_vocab_size argument is unused here. """
    def __init__(self, num_layers, d_model, num_heads, d_ffn,
                 input_vocab_size, target_vocab_size, pe_input,
                 pe_target, p_drop):
        super().__init__()
        self.embedding = Embedding(num_layers, d_model, num_heads, d_ffn, input_vocab_size+1,
                                   pe_input, p_drop)
        self.encoder = EncoderWithPretraining(num_layers, d_model, num_heads, d_ffn,
                                             pe_input, p_drop)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ffn,
                              target_vocab_size, pe_target, p_drop)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask,
            dec_padding_mask):
        inp = self.embedding(inp, training, enc_padding_mask)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attn_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attn_weights
