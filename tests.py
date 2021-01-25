import unittest
import tensorflow as tf
from model import *

class TestLayers(unittest.TestCase):

    def test_create_padding_mask(self):
        x = tf.constant([[5,9,0,0]], dtype=tf.int32)
        ans = tf.constant([[[[0,0,1,1]]]])
        pred = create_padding_mask(x)
        self.assertTrue( tf.experimental.numpy.allclose(pred, ans) )

    def test_create_look_ahead_mask(self):
        x = tf.constant([[8,2,0]])
        ans = tf.constant([[[[0,1,1],
                             [0,0,1],
                             [0,0,0]]]])
        pred = create_look_ahead_mask(tf.size(x))
        self.assertTrue( tf.experimental.numpy.allclose(pred, ans) )

    def testPositionalEncoding(self):
        posEnc = positionalEncoding(50, 512)
        self.assertTrue( posEnc.shape==(1,50,512) )

    def testFeedForwardNetwork(self):
        x = tf.random.normal((8,4,128))
        ffn = FeedForwardNetwork(d_model=128, d_ffn=1024)
        self.assertTrue(ffn(x).shape==x.shape)

    def testScaledDotProductAttention(self):
        bsz = 8
        d_model = 128
        num_heads = 3
        key_len = 12
        q_len = 4
        Q = tf.random.normal((bsz, num_heads, q_len, d_model))
        mask = tf.random.normal((1, 1, q_len, key_len))
        K = tf.random.normal((bsz, num_heads, key_len, d_model))
        V = tf.random.normal((bsz, num_heads, key_len, d_model))
        output, weights = scaledDotProductAttention(Q, K, V, mask)
        self.assertTrue(output.shape==(bsz, num_heads, q_len, d_model))
        self.assertTrue(weights.shape==(bsz, num_heads, q_len, key_len))

    def testMultiHeadAttention(self, ):
        bsz = 8
        d_model = 128
        num_heads = 3
        klen = 12
        qlen = 4
        with self.assertRaises(AssertionError):
            mha = MultiHeadAttention(d_model, num_heads)
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)
        x = tf.random.normal((bsz, klen, d_model))
        self.assertTrue(mha.split(x, bsz).shape == (bsz, num_heads, klen, d_model//num_heads))

        y = tf.random.normal((bsz, qlen, d_model))
        mask = tf.random.normal((1, 1, qlen, klen))
        output, weights = mha.call(y, x, x, mask)
        self.assertTrue(output.shape == y.shape)
        self.assertTrue(weights.shape == (bsz, num_heads, qlen, klen))

    def testEncoderLayer(self):
        bsz = 8
        d_model = 128
        num_heads = 4
        d_ffn = 1024
        klen = 12
        qlen = 12
        x = tf.random.normal((bsz, qlen, d_model))
        mask = tf.random.uniform( (bsz, 1, 1, qlen) )
        enc = EncoderLayer(d_model, num_heads, d_ffn)
        self.assertTrue(enc(x, training=True, mask=None).shape==x.shape)

    def testDecoderLayer(self):
        N, input_len, d_model, num_heads, d_ffn = 8, 13, 128, 8, 1024
        bsz = 8
        d_model = 128
        num_heads = 4
        d_ffn = 1024
        input_len = 12
        output_len = 7
        enc_input = tf.random.normal((bsz, input_len, d_model))
        dec_input = tf.random.normal((bsz, output_len, d_model))
        dec = DecoderLayer(d_model, num_heads, d_ffn)
        output, w1, w2 = dec(dec_input, enc_input, training=True, look_ahead_mask=None, padding_mask=None)
        self.assertTrue(output.shape==dec_input.shape)
        self.assertTrue( w1.shape==(bsz, num_heads, output_len, output_len) )
        self.assertTrue( w2.shape==(bsz, num_heads, output_len, input_len) )

    def testDecoder(self):
        bsz = 8
        d_model = 128
        num_heads = 4
        d_ffn = 1024
        num_layers = 2
        target_vocab_size=800
        input_vocab_size=500
        max_position=512
        input_len = 24
        target_len = 7
        sample_decoder = Decoder(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ffn=d_ffn,
            target_vocab_size=target_vocab_size, max_position=max_position
        )
        inp = tf.random.uniform((bsz, input_len), dtype=tf.int64, minval=0, maxval=input_vocab_size)
        targ = tf.random.uniform((bsz, target_len), dtype=tf.int64, minval=0, maxval=target_vocab_size)
        enc_input = tf.random.uniform((bsz, input_len, d_model))
        output, attn = sample_decoder(
            targ, enc_output=enc_input, training=False, look_ahead_mask=None, padding_mask=None
        )
        self.assertTrue( output.shape==(bsz, target_len, d_model) )
        self.assertTrue( attn['decoder_layer2_block2'].shape==(bsz, num_heads, target_len, input_len) )

    def testEncoder(self):
        bsz = 64
        d_model = 128
        num_heads = 4
        d_ffn = 1024
        num_layers = 2
        target_vocab_size=8500
        input_vocab_size=8500
        max_position=512
        input_len = 62
        target_len = 7
        sample_encoder = Encoder(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ffn=d_ffn,
             input_vocab_size=input_vocab_size, max_position=max_position
        )
        inp = tf.random.uniform((bsz, input_len), dtype=tf.int64, minval=0, maxval=input_vocab_size)
        sample_encoder_output = sample_encoder(inp, training=False, mask=None)
        self.assertTrue( sample_encoder_output.shape==(bsz, input_len, d_model) )

    def testTransformer(self):
        num_layers = 4
        d_model = 128
        num_heads = 8
        d_ffn = 1024
        input_vocab_size = 600
        target_vocab_size = 500
        pe_input = 60
        pe_target = 34
        p_drop = 0.1
        bsz = 16

        transformer = Transformer(num_layers, d_model, num_heads, d_ffn,
        input_vocab_size, target_vocab_size, pe_input, pe_target, p_drop)

        enc_padding_mask = tf.random.uniform( (bsz, 1, 1, pe_input) )
        look_ahead_mask = tf.random.uniform( (bsz, 1, pe_target, pe_target) )
        dec_padding_mask = tf.random.uniform( (bsz, 1, 1, pe_input) )

        inp = tf.random.uniform((bsz, pe_input), dtype=tf.int64, minval=0, maxval=input_vocab_size)
        tar = tf.random.uniform((bsz, pe_target), dtype=tf.int64, minval=0, maxval=target_vocab_size)
        out, weights = transformer(inp, tar, training=True, enc_padding_mask=enc_padding_mask,
                        look_ahead_mask=look_ahead_mask, dec_padding_mask=dec_padding_mask)
        self.assertTrue( out.shape==(bsz, pe_target, target_vocab_size) )
        self.assertTrue( weights['decoder_layer1_block1'].shape==(bsz, num_heads, pe_target, pe_target) )
        self.assertTrue( weights['decoder_layer1_block2'].shape==(bsz, num_heads, pe_target, pe_input) )

if __name__=="__main__":
    unittest.main()
