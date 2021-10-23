#!/usr/bin/env python

'''LSTM fasttext classifier '''

import sys
import argparse
import random
import time
import spacy
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

from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from dataParser import read, mergeCopEditions
from BaseModel import BaseModel 

class Bert(BaseModel):
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
            "command": "-tp",
            "refer": "--test_percentage",
            "default": 0.20,
            "action": None,
            "type": float,
            "help": "Percentage of the data that is used for the test set (default 0.20)"
        }
        ]

        super().__init__()
        self.name = "Bert"

    def vectorizer(self, samples, model):
        '''Turn sentence into embeddings, i.e. replace words by the fasttext word vector '''
        return np.array([ model.get_sentence_vector(sample) for sample in samples])

    # Create the model 
    def create_model(self, model): 
        loss_function = SparseCategoricalCrossentropy(from_logits=True)

        starter_learning_rate = 5e-5
        end_learning_rate = 5e-6
        decay_steps = 5
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            starter_learning_rate,
            decay_steps,
            end_learning_rate,
            power=0.5
        )

        optim = Adam(learning_rate=learning_rate_fn)

        model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
        return model

    # Train the model
    def train_model(self, model, tokens_train, tokens_test, Y_train_bin, Y_test_bin):
        '''Train the model here. Note the different settings you can experiment with!'''
        verbose = 2
        epochs = 50 #default 10
        batch_size = 16 #default 32

        validation_data = (tokens_test, Y_test_bin)

        # Early stopping: stop training when there are three consecutive epochs without improving
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        # Finally fit the model to our data
        model.fit(tokens_train, Y_train_bin, verbose=verbose, callbacks=[callback], epochs=epochs, batch_size=batch_size, validation_data=(tokens_test, Y_test_bin))
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

        # DEBUG
        print(results)

        # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w'))

    def perform_classification(self):
        pass

    def perform_cross_validation(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        prepared_data = np.array([article['headline'] for article in articles])

        # Transform string labels to one-hot encodings
        encoder = LabelBinarizer()
        labels = encoder.fit_transform([ article['political_orientation'] for article in articles])  # Use encoder.classes_ to find mapping back

        lm ="bert-base-cased"

        tokenizer = AutoTokenizer.from_pretrained(lm)

        model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6)

        # Perform KFold Cross Validation
        kfold = KFold(n_splits=2, shuffle=True)
        results = []
        n_fold = 1

        for train_index, test_index in kfold.split(prepared_data, labels):
            tokens_train = tokenizer(prepared_data[train_index].tolist(), padding=True, max_length=200,truncation=True, return_tensors="np").data
            tokens_test = tokenizer(prepared_data[test_index].tolist(), padding=True, max_length=200,truncation=True, return_tensors="np").data

            Y_train_bin = encoder.fit_transform(labels[train_index])
            Y_test_bin = encoder.fit_transform(labels[test_index])

            model = self.create_model(model) 
            model = self.train_model(model, tokens_train, tokens_test, Y_train_bin, Y_test_bin)
           
            scores = model.evaluate(tokens_test, Y_test_bin, verbose=2)
            results.append({n_fold: scores})
            n_fold += 1

        return results

if __name__ == "__main__":
    bert = Bert()
    
    results = bert.perform_cross_validation()

    # DEBUG
    #test_results = [{1: [0.307935893535614, 0.9006993174552917]}, {2: [0.1817009150981903, 0.9230769276618958]}]

    bert.write_run_to_file(vars(bert.args), results)
