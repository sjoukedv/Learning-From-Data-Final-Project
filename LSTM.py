#!/usr/bin/env python

'''LSTM fasttext classifier '''

import sys
import argparse
import random
import time
import spacy
import fasttext
import os
import json
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Activation, GlobalAveragePooling1D
from keras.initializers import Constant
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.preprocessing import LabelBinarizer

from dataParser import read, read_single, mergeCopEditions
from BaseModel import BaseModel 

class LSTM_Embeddings(BaseModel):
    def __init__(self):
        self.arguments = [
        { 
            "command": "-i",
            "refer": "--input_file",
            "default": "reviews.txt",
            "action": None,
            "type": str,
            "help": "Input file to learn from (default reviews.txt)"
        },
        { 
            "command": "-t",
            "refer": "--tfidf",
            "default": None,
            "type": None,
            "action": "store_true",
            "help": "Use the TF-IDF vectorizer instead of CountVectorizer"
        },
        { 
            "command": "-tp",
            "refer": "--test_percentage",
            "default": 0.20,
            "action": None,
            "type": float,
            "help": "Percentage of the data that is used for the test set (default 0.20)"
        }
        ]

        super().__init__()
        self.name = "LSTM"

    def perform_classification(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        prepared_data = [ article['headline'] for article in articles]
        
        # Transform words to fasttext embeddings
        # link to file https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
        fasttext_model = fasttext.load_model('cc.en.300.bin')
        embedded_data = self.vectorizer(prepared_data, fasttext_model)
        labels = [ article['political_orientation'] for article in articles]

        # Transform string labels to one-hot encodings
        encoder = LabelBinarizer()
        labels = encoder.fit_transform([ article['political_orientation'] for article in articles])  # Use encoder.classes_ to find mapping back

        # create and train model
        model = self.create_model(embedded_data, labels)
        model = self.train_model(model, embedded_data, labels)

        # read and convert test data
        test_articles = read_single(self.args.test_file)
        test_prepared_data = [ article['headline'] for article in test_articles]
        test_embedded_data = self.vectorizer(test_prepared_data, fasttext_model)
        true_articles =  [ article['political_orientation'] for article in test_articles]
           
        print(model.evaluate(test_embedded_data, true_articles, verbose=1))


    def vectorizer(self, samples, model):
        '''Turn sentence into embeddings, i.e. replace words by the fasttext word vector '''
        # return np.array([ [model.get_word_vector(word) for word in sample] for sample in samples])
        return np.array([ model.get_sentence_vector(sample) for sample in samples])

    def create_model(self, X_train, Y_train): 
        model = Sequential()
        model.add(Embedding(300, 300, trainable=False))
        model.add(LSTM(units=128, dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
        return model

    def train_model(self, model, X_train, Y_train):
        '''Train the model here. Note the different settings you can experiment with!'''
        # Potentially change these to cmd line args again
        # And yes, don't be afraid to experiment!
        verbose = 1
        epochs = 5 #default 10
        batch_size = 32 #default 32
        # 10 percent of the training data we use to keep track of our training process
        # Use it to prevent overfitting!
        validation_split = 0.1

        early_acc = EarlyStopping(monitor='accuracy', patience=1)

        # Finally fit the model to our data
        model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, batch_size=batch_size, callbacks=[early_acc])
        return model

    def write_run_to_file(self, parameters, results):
        res_dir = 'results/' + self.name
        # make sure (sub)directory exists
        os.makedirs(res_dir, exist_ok=True)

        # retrieve version based on number of files in directory
        path, dirs, files = next(os.walk(res_dir))
        version = len(files)

        result = {
            'parameters' : parameters,
            'results' : results
            }

        print(results)
        # convert array to list
        # TODO fix this loop
        for res in results:
            if hasattr(results[res], "__len__"):
                result['results'][res] = results[res].tolist()

        # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w'))

        

    def perform_cross_validation(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        prepared_data = [ article['headline'] for article in articles]

        # Transform words to fasttext embeddings
        # link to file https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
        fasttext_model = fasttext.load_model('cc.en.300.bin')
        embedded_data = self.vectorizer(prepared_data, fasttext_model)

        # Transform string labels to one-hot encodings
        encoder = LabelBinarizer()
        labels = encoder.fit_transform([ article['political_orientation'] for article in articles])  # Use encoder.classes_ to find mapping back

        # Perform KFold Cross Validation
        kfold = KFold(n_splits=3, shuffle=True)
        results = []
        n_fold = 1
        for train, test in kfold.split(embedded_data, labels):
            model = self.create_model(embedded_data[train], labels[train])
            model = self.train_model(model, embedded_data[train], labels[train])
           
            scores = model.evaluate(embedded_data[test], labels[test], verbose=0)
            results.append({n_fold: scores})
            n_fold += 1

        # TODO optional GridSearch for values
        return results

if __name__ == "__main__":
    lstm = LSTM_Embeddings()

    # run test
    if lstm.args.test_file:
        lstm.perform_classification()
    # run dev
    else:
        results = lstm.perform_cross_validation()
        lstm.write_run_to_file(vars(lstm.args), results)
