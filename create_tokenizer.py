import os
import time
import sentencepiece as spm
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string('text_file', default='', help='text file to train tokenizer on')
flags.DEFINE_string('prefix', default='', help='model prefix name')
flags.DEFINE_integer('vocab_size', default=2048, help='vocab size')
flags.DEFINE_boolean('lowercase', default=True, help='Whether to lowercase or not.')

# Train a sentencepiece model
def build_sentencepiece_model(input_file, model_prefix, vocab_size, lowercase):
    if not os.path.isfile(input_file):
        raise FileExistsError("Input text file does not exist.")
    if not input_file.endswith('.txt'):
        raise ValueError("Input text file must end with .txt extension.")

    normalization_rule = 'nfkc_cf' if lowercase else 'nfkc'

    print("\n\nTraining SentencePiece on file: {input_file}...\n\n")
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
    print(f"""\n\nFinished training. Time: {time.time()-start:.2f}s.
     Model protobuf saved to: {model_prefix}.model.
     Vocab size: {vocab_size}.\n\n""")
    return f"{model_prefix}.model"

def main(argv):
    build_sentencepiece_model(FLAGS.text_file, FLAGS.prefix, FLAGS.vocab_size, FLAGS.lowercase)

if __name__=="__main__":
    app.run(main)
