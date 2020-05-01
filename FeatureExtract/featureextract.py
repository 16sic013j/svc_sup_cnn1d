from keras import Model
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tqdm import tqdm

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

    print("tokenising..")
    tokenizer = Tokenizer(num_words=max_words)
    model = load_model(modelpath)
    storefeat = []
    for val in tqdm(data):
        tokenizer.fit_on_texts(val)
        sequences = tokenizer.texts_to_sequences(val)
        val = pad_sequences(sequences, maxlen=maxsequence)
        features = model.predict(val, verbose=0)
        storefeat.append(features)

    storefeat = np.asarray(storefeat)
    np.save(featurefilepath, storefeat)
    print("Feature extracted.")
