import time
import numpy as np
import tensorflow as tf

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def loss_fn(real, logits, mask):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(real, logits)
    return tf.reduce_mean(tf.boolean_mask(loss, mask))

def accuracy_fn(real, logits, mask):
    preds = tf.cast(tf.argmax(logits, -1), dtype=real.dtype)
    accuracy = tf.cast(tf.equal(real, preds), tf.float32)
    return tf.reduce_mean(tf.boolean_mask(accuracy, mask))

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps, decay_steps, **kwargs):
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
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
