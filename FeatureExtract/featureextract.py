from keras import Model
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from FeatureExtract.trainmodel import trainingmodel
import numpy as np


def extractfeature(preprocessedfile, labelfile, classlen, glovefile, uniquetokenfile, modelpath, featurefilepath,maxsequence,
                  max_words,
                  embed_dim,
                  valid_split):

    trainingmodel(preprocessedfile, labelfile, classlen, glovefile, uniquetokenfile, modelpath, maxsequence,
                  max_words,
                  embed_dim,
                  valid_split)


    data = []
    # get files
    preprocessread = open(preprocessedfile, 'r')
    for line in preprocessread.readlines():
        data.append(line)

    print("tokenising..")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    # Padding of data insto sequence
    print("Padding sequences...")
    data = pad_sequences(sequences, maxlen=maxsequence)
    print('Shape of Data Tensor:', data.shape)

    print("Extracting features...")
    model = load_model(modelpath)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
    intermediate_layer_model.summary()
    feauture_engg_data = intermediate_layer_model.predict(data)
    np.save(featurefilepath, feauture_engg_data)
    print("Feature extracted.")
