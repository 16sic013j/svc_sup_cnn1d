from keras import Model
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from FeatureExtract.trainmodel import trainingmodel
import numpy as np


def extractfeature(preprocessedfile,
                   featurefilepath,
                   modelpath,
                   maxsequence,
                   max_words):

    data = []
    # get files
    preprocessread = open(preprocessedfile, 'r')
    for line in preprocessread.readlines():
        data.append(line)

    print(len(data))
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
    features = model.predict(data, verbose=1)
    np.save(featurefilepath, features)
    print("Feature extracted.")
