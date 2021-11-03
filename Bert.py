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

from dataParser import read_articles, read_single
from BaseModel import BaseModel 

class Bert(BaseModel):
    def __init__(self):
        self.arguments = [
        { 
            "command": "-test",
            "refer": "--test",
            "default": False,
            "action": "store_true",
            "help": "Run predictions on test set (otherwise uses dev set)"
        },
        { 
            "command": "-load",
            "refer": "--load_model",
            "default": False,
            "action": "store_true",
            "help": "Load existing model or perform training"
        },
        {
            "command": "-cop",
            "refer": "--cop",
            "default": None,
            "action": None,
            "type:": str,
            "help": "Path to single COP edition to test (e.g. data/COP25.filt3.sub.json)"
        },
        {
            "command": "-batch_size",
            "refer": "--batch_size",
            "default": 16,
            "action": None,
            "type": int,
            "help": "Batch size"
        },
        {
            "command": "-epochs",
            "refer": "--epochs",
            "default": 50,
            "action": None,
            "type": int,
            "help": "Epochs"
        },
        {
            "command": "-max_length",
            "refer": "--max_length",
            "default": 200,
            "action": None,
            "type": int,
            "help": "Max length"
        },
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

        validation_data = (tokens_test, Y_test_bin)

        # Early stopping: stop training when there are three consecutive epochs without improving
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        # Finally fit the model to our data
        model.fit(tokens_train, Y_train_bin, verbose=verbose, callbacks=[callback], epochs=self.args.epochs, batch_size=self.args.batch_size, validation_data=(tokens_test, Y_test_bin))
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

    def perform_classification(self, model, X, Y, tokenizer, encoder):
        tokens_X = tokenizer(X, padding=True, max_length=self.args.max_length,truncation=True, return_tensors="np").data

        labels_Y = encoder.fit_transform(Y)
        
        scores = model.evaluate(tokens_X, labels_Y, verbose=2)
        print(f'test loss: {scores[0]}, test acc:{scores[1]}')

        return results

    def perform_cross_validation(self):
        pass

if __name__ == "__main__":
    bert = Bert()

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles()
    
    if bert.args.undersample:
        X_train, Y_train = bert.under_sample_training_data(X_train, Y_train)

    encoder = LabelBinarizer()

    lm ="bert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(lm)

    if bert.args.load_model:
        model = bert.load_keras_model()
    else:
        # train
        print('Training model')

        model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6)

        tokens_train = tokenizer(X_train, padding=True, max_length=200,truncation=True, return_tensors="np").data
        tokens_test = tokenizer(X_dev, padding=True, max_length=200,truncation=True, return_tensors="np").data

        labels_train = encoder.fit_transform(Y_train)
        labels_test = encoder.fit_transform(Y_dev)

        model = bert.create_model(model) 
        model = bert.train_model(model, tokens_train, tokens_test, labels_train, labels_test)

        # save model 
        bert.save_keras_model(model)
    
    # run test
    if bert.args.test and not bert.args.cop:
        print('Using best estimator on Test set')
        results = bert.perform_classification(model, X_test, Y_test, tokenizer, encoder)
    # run dev
    elif not bert.args.cop:
        print('Using best estimator on Dev set')
        results = bert.perform_classification(model, X_dev, Y_dev, tokenizer, encoder)
    
    # test model with COP25 edition
    if bert.args.cop:
        print(f'Predicting {bert.args.cop}')
        X_cop, Y_cop = read_single(bert.args.cop)
        results = bert.perform_classification(model, X_cop, Y_cop, tokenizer, encoder)

    bert.write_run_to_file(vars(bert.args), results)
