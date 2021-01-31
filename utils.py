import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from difflib import SequenceMatcher


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def print_bar(step, tot, diff, loss):
    if tot is None:
        iter_message = f"Iteration {step+1:02d}/unknown:"
        time_message = f"{1/diff:.2f} it/s."
        loss_message = f"Loss: {loss:.3f}"
        print(iter_message, time_message, loss_message, end='\r')
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
    end = '\r' if step<tot-1 else '\n'
    print(iter_message, bar, time_message, loss_message, end=end)

def predict(input_sentence, inp_tokenizer, model, temp=1, max_length=100):
    inp = tf.concat([[1], inp_tokenizer.tokenize(input_sentence), [2]], 0)[tf.newaxis, :]
    tar = tf.constant([1], dtype=inp.dtype)[tf.newaxis, :]
    for i in range(max_length):
        enc_mask, comb_mask, dec_mask = create_masks(inp, tar)
        preds, weights = model(inp, tar, training=False, enc_padding_mask=enc_mask,
                           look_ahead_mask=comb_mask, dec_padding_mask=dec_mask)
        next_token = tf.cast(tf.random.categorical(preds[:,-1]/temp, 1), tar.dtype)
        if tf.reduce_all(tf.equal(next_token, 2)):
            break
        tar = tf.concat([tar, next_token], axis=1)
    return tar, weights

def plot_attn_weights(inp_text, tar_seq, attn_weights,
                      inp_tokenizer, tar_tokenizer):
    fig = plt.figure(figsize=(16, 16))

    inp_seq = inp_tokenizer.tokenize(inp_text)
    tar_seq = tf.squeeze(tar_seq, 0)
    inp_seq = [x.numpy().decode() for x in inp_tokenizer.id_to_string(inp_seq)]
    tar_seq = [x.numpy().decode() for x in tar_tokenizer.id_to_string(tar_seq)]
    attn_weights = tf.squeeze(attn_weights, 0)
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 2, head+1)
        ax.matshow(attn_weights[head], cmap='viridis')
        fontdict = {'fontsize': 14}
        ax.set_xticks(range(len(inp_seq)+2))
        ax.set_yticks(range(len(tar_seq)))
        ax.set_xticklabels(['<sos>']+inp_seq+['<eos>'], fontdict=fontdict,
                           rotation=45)
        ax.set_yticklabels(tar_seq[1:]+['<eos>'], fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head+1))

    plt.tight_layout()
    plt.show()


def batch_random_predict(input_text, inp_tokenizer, model,
                         temp=1.0, n_samples=8, max_length=75):

    inp = tf.concat([[1], inp_tokenizer.tokenize(input_text), [2]], 0)
    inp = tf.repeat(inp[tf.newaxis, :], n_samples, axis=0)

    tar = tf.ones((n_samples,1), dtype=inp.dtype)
    mask = tf.ones((n_samples, 1), dtype=tf.bool) # for ended sentences

    for _ in range(max_length):
        enc_mask, comb_mask, dec_mask = create_masks(inp, tar)
        logits, weights = model(inp, tar, training=False,
                               enc_padding_mask=enc_mask,
                               look_ahead_mask=comb_mask,
                               dec_padding_mask=dec_mask)

        logits = logits[:,-1,:]/tf.cast(temp, tf.float32)
        next_tokens = tf.cast(tf.random.categorical(logits, 1), inp.dtype)

        # replace any already finished sentences with padding 4 next token
        next_tokens = tf.where(mask, next_tokens, 0)

        # update mask. to stay true, it must:
        # 1) already be true
        # 2) not currently be an end of sentence token
        mask = tf.logical_and(mask, tf.not_equal(next_tokens, 2))
        tar = tf.concat([tar, next_tokens], axis=1)
        if not tf.reduce_any(mask):
            break
    return tar, weights

def find_most_similar(arr, topK=0):
    """ For each row, it computes the mean similarity score with all other
    rows and stores it in sims. Then it chooses the row with highest mean
    and returns the array at that index."""
    sims = []
    N = arr.shape[0]
    for i in range(N):
        sims.append(np.mean([SequenceMatcher(a=arr[i], b=arr[j]).ratio()
                  for j in range(N) if j!=i]))
    if topK:
        ids = np.argpartition(sims, -topK)[-topK:]
        return sorted(ids, key=lambda x: sims[x])
    return np.argmax(sims)

def minimum_bayes_risk_predict(input_text, inp_tokenizer, model, topK=0,
                               temp=1.0, max_length=75, n_samples=8):
    tar_batch, weights_batch = batch_random_predict(
        input_text=input_text,
        inp_tokenizer=inp_tokenizer,
        model=model,
        temp=temp,
        max_length=max_length,
        n_samples=n_samples
    )
    tar_batch = tar_batch.numpy()
    idx = find_most_similar(tar_batch)
    return tar_batch[idx], weights_batch[idx]
