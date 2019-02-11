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
    seq_in = seg_text[i:i+seq_len]
    data_x.append([word_2_int[word] for word in seq_in])

n_samples = len(data_x)

print('total samples: ' + str(n_samples))

# Start generate the new lyrics

model = load_model('trained_model.h5')

out = []

pattern = data_x[(np.random.randint(0, n_samples-1))]

pattern[len(pattern)-1] = 4410
pattern[len(pattern)-2] = 1968
pattern[len(pattern)-3] = 4440
pattern[len(pattern)-4] = 1722
pattern[len(pattern)-5] = 3943

init = ''.join([int_2_word[w] for w in pattern][95:])

for i in range(88):

    x = np.reshape(pattern, (1, len(pattern)))

    pred = model.predict(x, verbose=0)[0]

    index = np.argmax(pred)

    word = int_2_word[index]

    out.append(word)

    pattern.append(index)

    pattern = pattern[1:len(pattern)]

print('')

print(init)

for i in range(0, len(out)-5, 5):
    print(''.join(out[i:i+5]))




