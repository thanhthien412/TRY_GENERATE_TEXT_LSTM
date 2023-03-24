import numpy as np
import random
import sys
from keras.models import load_model
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str,required=True,
help="pre trained model")
ap.add_argument("-txt", "--txt", type=str,required=True,
help="text file")
ap.add_argument("-seq", "--seq", type=int,default=60,
help="sequence length")
ap.add_argument("-num", "--number", type=int,default=60,
help="number of word want to generate")
args = vars(ap.parse_args())
seq_length = args['seq'] 
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds) #exp of log (x), isn't this same as x??
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1) 
    return np.argmax(probas)

raw_text = open(args['txt'], 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

raw_text = ''.join(c for c in raw_text if not c.isdigit())
chars = sorted(list(set(raw_text)))


char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


n_chars = len(raw_text)
n_vocab = len(chars)

model=load_model(args['model'])

start_index = random.randint(0, n_chars - seq_length - 1)

generated = ''
sentence = raw_text[start_index: start_index + seq_length]
generated += sentence

for i in range(args['number']):
    x_pred = np.zeros((1, seq_length, n_vocab))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_to_int[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)
    next_char = int_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()

print()