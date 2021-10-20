#!/usr/bin/env python

'''LSTM fasttext classifier '''

import sys
import argparse
import random
import time
import spacy
import fasttext
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Activation
from keras.initializers import Constant
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.preprocessing import LabelBinarizer

from dataParser import read, mergeCopEditions
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

    def split_data(self, X_full, Y_full, test_percentage):
        ## This method is responsible for splitting the data into test and training sets, based on the percentage. 
        ## The two training and two test sets are returned. 
        split_point = int((1.0 - test_percentage)*len(X_full))

        X_train = X_full[:split_point]
        Y_train = Y_full[:split_point]
        X_test = X_full[split_point:]
        Y_test = Y_full[split_point:]
        return X_train, Y_train, X_test, Y_test

    def perform_classification(self):
        pass

    def vectorizer(self, samples, model):
        '''Turn sentence into embeddings, i.e. replace words by the fasttext word vector '''
        # return np.array([ [model.get_word_vector(word) for word in sample] for sample in samples])
        return np.array([ model.get_sentence_vector(sample)for sample in samples])

    def create_model(self, X_train, Y_train): 
        print('Y_shape', Y_train.shape)
        print('X_shape', X_train.shape)
        model = Sequential()
        model.add(Embedding(100, 100, trainable=False))
        
        model.add(LSTM(units=32, dropout=0.2))
     
        model.add(Dense(1, activation='softmax'))

        # TODO remove debug
        print(model.summary())

        model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
        return model

    def train_model(self, model, X_train, Y_train):
        '''Train the model here. Note the different settings you can experiment with!'''
        # Potentially change these to cmd line args again
        # And yes, don't be afraid to experiment!
        verbose = 1
        epochs = 10 #default 10
        batch_size = 32 #default 32
        # 10 percent of the training data we use to keep track of our training process
        # Use it to prevent overfitting!
        validation_split = 0.1
        # Finally fit the model to our data
        model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return model
        

    def perform_cross_validation(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        prepared_data = [ article['headline'] for article in articles]

        # TODO use a predefined set of vectors
        # https://ppasumarthi-69210.medium.com/word-embeddings-in-keras-be6bb3092831
        # Transform words to fasttext embeddings
        np.savetxt('fasttext_raw.csv', prepared_data, delimiter=',', fmt='%s')
        fasttext_model = fasttext.train_unsupervised('fasttext_raw.csv')
        embedded_data = self.vectorizer(prepared_data, fasttext_model)

        # Transform string labels to one-hot encodings
        encoder = LabelBinarizer()
        labels = encoder.fit_transform([ article['political_orientation'] for article in articles])  # Use encoder.classes_ to find mapping back

        # Perform KFold Cross Validation
        kfold = KFold(n_splits=3, shuffle=True)
        results = []
        for train, test in kfold.split(embedded_data, labels):
            
            model = self.create_model(embedded_data[train], labels[train])
            model = self.train_model(model, embedded_data[train], labels[train])
           

            scores = model.evaluate(embedded_data[test], labels[test], verbose=0)
            results.append(scores)

            # TODO remove
            sys.exit(0)

        # TODO optional GridSearch for values
        return results

if __name__ == "__main__":
    lstm = LSTM_Embeddings()
    results = lstm.perform_cross_validation()
    lstm.write_run_to_file(vars(lstm.args), results)
