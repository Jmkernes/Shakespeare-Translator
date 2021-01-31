import time
import numpy as np
import tensorflow as tf

def positionalEncoding(p_max, d_model, min_freq=1e-4):
    freqs = min_freq**(2.0*(np.arange(d_model)//2)/d_model)
    p = np.arange(p_max)
    matrix = p[:, np.newaxis]*freqs[np.newaxis, :]
    matrix[:,::2] = tf.math.sin(matrix[:,::2])
    matrix[:,1::2] = tf.math.cos(matrix[:,1::2])
    pos_encoding = tf.expand_dims(matrix, 0)
    pos_encoding = tf.cast(pos_encoding, tf.float32)
    return pos_encoding

def scaledDotProductAttention(Q, K, V, mask=None):
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    logits = tf.matmul(Q, K, transpose_b=True)
    scaled_logits = logits/tf.math.sqrt(dk)
    if mask is not None:
        scaled_logits += -1e9*mask
    weights = tf.nn.softmax(scaled_logits, axis=-1)
    res = tf.matmul(weights, V)
    return res, weights

class MultiHeadAttention(tf.keras.layers.Layer):
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

def FeedForwardNetwork(d_model, d_ffn):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(d_ffn, activation=tf.keras.activations.gelu),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, p_drop=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ffn)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(p_drop)
        self.dropout2 = tf.keras.layers.Dropout(p_drop)

    def call(self, x, training, mask):
        attn, weights = self.mha(x, x, x, mask)
        attn = self.dropout1(attn, training=training)
        out1 = self.layernorm1(x+attn)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1+ffn_output)
        return out2, weights

class Encoder(tf.keras.layers.Layer):
    """ In contrast to layer used in the main model, this does not use an embedding layer """
    def __init__(self, num_layers, d_model, num_heads, d_ffn, max_position=512, dropout_rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positionalEncoding(max_position, d_model)
        self.encoders = [EncoderLayer(d_model, num_heads, d_ffn, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_weights = {}
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:,:seq_len, :]
        x = self.dropout(x, training=training)
        for i, encode in enumerate(self.encoders):
            x, weights = encode(x, training, mask)
            attn_weights[f'Encoder_layer_{i+1}'] = weights
        return x, attn_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, vocab_size, max_position=512, dropout_rate=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.W_emb = self.add_weight(shape=(vocab_size, d_model), initializer='glorot_normal',
                                    trainable=True, name='Embeddings')
        self.encoder = Encoder(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ffn=d_ffn,
            max_position=max_position, dropout_rate=dropout_rate)

    def call(self, x, training, mask):
        x = tf.nn.embedding_lookup(self.W_emb, x)
        x, attn_weights = self.encoder(x, training, mask)
        x = tf.matmul(x, self.W_emb, transpose_b=True)
        return x, attn_weights

class Embedding(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, vocab_size, max_position=512, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.W_emb = self.add_weight(shape=(vocab_size, d_model), initializer='glorot_normal',
                                    trainable=True, name='Embeddings')
        self.encoder = Encoder(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ffn=d_ffn,
            max_position=max_position, dropout_rate=dropout_rate)

    def call(self, x, training, mask):
        x = tf.nn.embedding_lookup(self.W_emb, x)
        x, attn_weights = self.encoder(x, training, mask)
        return x
