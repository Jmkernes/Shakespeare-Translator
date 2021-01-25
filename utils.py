import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, seq_len, seq_len)

# def create_masks(inp, tar):
#     enc_padding_mask = create_padding_mask(inp)
#     dec_padding_mask = create_padding_mask(inp)
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#     return enc_padding_mask, combined_mask, dec_padding_mask

def print_bar(step, tot, diff, loss):
    if tot is None:
        iter_message = f"Iteration {step+1:02d}/unknown:"
        time_message = f"{1/diff:.2f} it/s."
        loss_message = f"Loss: {loss:.3f}"
        if step%100==0:
            print(iter_message, time_message, loss_message, end='\n')
        return
    num_eq = int(10*(step+1)/tot)
    num_pd = 10-num_eq
    bar = '['+'='*num_eq+'>'+'.'*num_pd+']'
    time_left = (tot-step)*diff
    m = int(time_left)//60
    s = int(time_left)%60
    iter_message = f"Iteration {step+1:02d}/{tot}:"
    time_message = f"{1/diff:.2f} it/s. Est: {m:02d}m {s:02d}s"
    loss_message = f"Loss: {loss:.3f}"
#    end = '\r' if step<tot-1 else '\n'
    end = '\r' if step%100 else '\n'
    print(iter_message, bar, time_message, loss_message, end=end)


# def evaluate(inp_sentence):
#     start_token = [1]
#     end_token = [2]
#     inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
#     encoder_input = tf.expand_dims(inp_sentence, 0)
#
#     # as the target is english, the first word to the transformer should be the
#     # english start token.
#     decoder_input = [tokenizer_en.vocab_size]
#     output = tf.expand_dims(decoder_input, 0)
#
#     for i in range(MAX_LENGTH):
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
#             encoder_input, output)
#
#         # predictions.shape == (batch_size, seq_len, vocab_size)
#         predictions, attention_weights = transformer(
#             encoder_input, output, False, enc_padding_mask,
#             combined_mask, dec_padding_mask)
#
#         # select the last word from the seq_len dimension
#         predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
#
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
#
#         # return the result if the predicted_id is equal to the end token
#         if predicted_id == tokenizer_en.vocab_size+1:
#             return tf.squeeze(output, axis=0), attention_weights
#
#         # concatentate the predicted_id to the output which is given to the decoder
#         # as its input.
#         output = tf.concat([output, predicted_id], axis=-1)
#     return tf.squeeze(output, axis=0), attention_weights

# def plot_attention_weights(attention, sentence, result, layer):
#     fig = plt.figure(figsize=(16, 8))
#     sentence = tokenizer_pt.encode(sentence)
#     attention = tf.squeeze(attention[layer], axis=0)
#     for head in range(attention.shape[0]):
#         ax = fig.add_subplot(2, 4, head+1)
#         ax.matshow(attention[head][:-1, :], cmap='viridis')
#         fontdict = {'fontsize': 10}
#         ax.set_xticks(range(len(sentence)+2))
#         ax.set_yticks(range(len(result)))
#         ax.set_ylim(len(result)-1.5, -0.5)
#         ax.set_xticklabels(
#             ['<sos>']+[tokenizer_pt.decode([i]) for i in sentence]+['<eos>'],
#             fontdict=fontdict, rotation=90)
#         ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
#                             if i < tokenizer_en.vocab_size],
#                            fontdict=fontdict)
#         ax.set_xlabel('Head {}'.format(head+1))
#     plt.tight_layout()
#     plt.show()

# def translate(sentence, plot=''):
#     result, attention_weights = evaluate(sentence)
#     predicted_sentence = tokenizer_en.decode([i for i in result
#                                             if i < tokenizer_en.vocab_size])
#     print('Input: {}'.format(sentence))
#     print('Predicted translation: {}'.format(predicted_sentence))
#     if plot:
#         plot_attention_weights(attention_weights, sentence, result, plot)
