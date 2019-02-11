from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import jieba

f = open('jay.txt', 'r')

text = f.read().replace('\n', '').replace(" ", "")

text = " ".join(text.split())

f.close()

seg_text = []

cut_text = jieba.cut(text)

for w in cut_text:
    seg_text.append(w)

vocab = sorted(list(set(seg_text)))

word_2_int = dict((w, i) for i, w in enumerate(vocab))
int_2_word = dict((i, w) for i, w in enumerate(vocab))

n_words = len(seg_text)
n_vocab = len(vocab)

print('total words: ' + str(n_words))
print('total vocab: ' + str(n_vocab))

seq_len = 100

data_x = []
data_y = []

for i in range(0, n_words-seq_len, 1):
    seq_in = seg_text[i:i+seq_len+1]
    data_x.append([word_2_int[word] for word in seq_in])

np.random.shuffle(data_x)

n_samples = len(data_x)

print('total samples: ' + str(n_samples))

for i in range(n_samples):
    data_y.append(data_x[i][seq_len])
    data_x[i] = data_x[i][:seq_len]

X = np.reshape(data_x, (n_samples, seq_len))
Y = np_utils.to_categorical(data_y)

# build network

model = Sequential()
model.add(Embedding(n_vocab, 512, input_length=seq_len))
model.add(LSTM(512, input_shape=(seq_len, 512), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(1024))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation="softmax"))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam)

model.fit(X, Y, batch_size=100, epochs=30, verbose=1)

model_path = 'trained_model.h5'

model.save(model_path)



