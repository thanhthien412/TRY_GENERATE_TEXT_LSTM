from keras.models import Model
from keras.layers import Dense, LSTM, Flatten,Input,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-seq", "--seq", type=int,default=60,
help="sequence length")
ap.add_argument("-step", "--step", type=int,default=10,
help="step to pass when seperate data")
ap.add_argument("-epoch", "--epoch", type=int,default=60,
help="number of epoch")
ap.add_argument("-batch", "--batch", type=int,default=128,
help="number of batch")
ap.add_argument("-lr", "--lr", type=float,default=0.001,
help="learning rate")
ap.add_argument("-txt", "--txt", type=str,required=True,
help="text file")
args = vars(ap.parse_args())
seq_length = args['seq'] 
step = args['step']

raw_text = open(args['txt'], 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
raw_text = ''.join(c for c in raw_text if not c.isdigit())
chars = sorted(list(set(raw_text)))


char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


n_chars = len(raw_text)
n_vocab = len(chars)


def build():
    input=Input(shape=(seq_length,n_vocab))
    output=LSTM(128,return_sequences=True)(input)
    output=Dropout(0.4)(output)
    output=LSTM(128)(input)
    output=Dropout(0.3)(output)
    output=Dense(n_vocab,activation='softmax')(output)
    optimizer = RMSprop(learning_rate=args['lr'])
    model =Model(input,output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return  model

sentences = []   
next_chars = []  
 
for i in range(0, n_chars - seq_length, step):  
    sentences.append(raw_text[i: i + seq_length])
    next_chars.append(raw_text[i + seq_length])  
n_patterns = len(sentences)    

#create x,y to train 
x = np.zeros((len(sentences), seq_length, n_vocab), dtype=np.bool)
y = np.zeros((len(sentences), n_vocab), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1
    
model=build()

history = model.fit(x, y,
          batch_size=args['batch'],
          epochs=args["epoch"])

model.save('weights_{}epochs.h5'.format(args['epoch']))