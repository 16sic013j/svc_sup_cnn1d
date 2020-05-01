import os

import numpy as np

os.environ['KERAS_BACKEND'] = 'theano'  # Why theano why not
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint


def trainingmodel(preprocessedfile,
                  labelfile,
                  classlen,
                  glovefile,
                  uniquetokenfile,
                  modelpath,
                  maxsequence,
                  max_words,
                  embed_dim,
                  valid_split):
    print("Training model...")
    preprocessed_arr = []
    labelarr = []

    # get files
    print("getting files..")
    preprocessread = open(preprocessedfile, 'r')
    labelread = open(labelfile, 'r')
    for line in preprocessread.readlines():
        preprocessed_arr.append(line)
    for line in labelread.readlines():
        labelarr.append(line)


    print("tokenising..")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(preprocessed_arr)
    sequences = tokenizer.texts_to_sequences(preprocessed_arr)
    word_index = tokenizer.word_index

    with open(uniquetokenfile, 'w') as file:
        for key, value in word_index.items():
            file.write(str(key) + ":" + str(value) + '\n')
    print('Number of Unique Tokens', str(len(word_index)))

    # Padding of data insto sequence
    print("Padding sequences...")
    data = pad_sequences(sequences, maxlen=maxsequence)
    labels = to_categorical(np.asarray(labelarr))
    print('Shape of Data Tensor:', data.shape)
    print('Shape of Label Tensor:', labels.shape)

    # Split train and validation
    print("spliting train and validation")
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(valid_split * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    # Prepare embedding layer
    print("prepraring embedded layer...")
    embeddings_index = {}
    f = open(glovefile, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,
                                embed_dim, weights=[embedding_matrix],
                                input_length=maxsequence, trainable=True)

    # Prepare model
    print("preparing model..")
    sequence_input = Input(shape=(maxsequence,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(classlen, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()
    cp = ModelCheckpoint(modelpath, monitor='val_acc', verbose=1, save_best_only=True)

    print("running model..")
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=15, callbacks=[cp])

    print("Model trained...")
