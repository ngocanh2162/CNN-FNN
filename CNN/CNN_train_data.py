# Chạy chương trình trên Colab thì bỏ cmt 3 dòng dưới
# !pip3 install numpy==1.16.2
# import numpy as np
# print(np.__version__)

from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from gensim.models import word2vec
from os.path import join, exists, split
from keras.layers import  Dropout, Input,  Embedding
import os
import pickle

def load_data():
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)
        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")
        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
        return x_train, y_train, x_test, y_test, vocabulary_inv

def train_word2vec(sentence_matrix, vocabulary_inv,num_features=300, min_word_count=1, context=10):
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        num_workers = 2  
        downsampling = 1e-3  
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)
        embedding_model.init_sims(replace=True)
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights

embedding_dim = 50
dropout_prob = (0.5, 0.8)
sequence_length = 400
max_words = 5000
min_word_count = 1
context = 10

print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data()

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
print("x_train static shape:", x_train.shape)
print("x_test static shape:", x_test.shape)
input_shape = (sequence_length, embedding_dim)
model_input = Input(shape=input_shape)
z = model_input
z = Dropout(dropout_prob[0])(z)

pickle.dump(y_train, open('Trained Data/imdb/y_train.pkl', 'wb'))
pickle.dump(y_test, open('Trained Data/imdb/y_test.pkl', 'wb'))
pickle.dump(x_train, open('Trained Data/imdb/x_train.pkl', 'wb'))
pickle.dump(x_test, open('Trained Data/imdb/x_test.pkl', 'wb'))
pickle.dump(z, open('Trained Data/imdb/z.pkl', 'wb'))
pickle.dump(model_input, open('Trained Data/imdb/inpu.pkl', 'wb'))