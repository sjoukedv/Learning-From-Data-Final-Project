#!/usr/bin/env python

'''LSTM fasttext classifier '''

import fasttext
import os
import json
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

from dataParser import read_articles, read_single
from BaseModel import BaseModel 

class LSTM_Embeddings(BaseModel):
    def __init__(self):
        self.arguments = [
        {
            "command": "-cv",
            "refer": "--cv",
            "default": 3,
            "action": None,
            "type:": int,
            "help": "Determines the cross-validation splitting strategy"
        },
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
        ]

        super().__init__()
        self.name = "LSTM"

    def vectorizer(self, samples, model):
        '''Turn sentence into embeddings, i.e. replace words by the fasttext word vector '''
        return np.array([ model.get_sentence_vector(sample) for sample in samples])

    def create_model(self, X_train, Y_train): 
        model = Sequential()

        # add layers
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
        model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, batch_size=batch_size, callbacks=[early_acc], validation_split=validation_split)
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

      # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w')) 

    def perform_classification(self, model, X, Y, fasttext_model, encoder):
        test_prepared_data = X
        test_embedded_data = self.vectorizer(test_prepared_data, fasttext_model)
        true_articles = encoder.fit_transform(Y)
  
        scores = model.evaluate(test_embedded_data, true_articles, verbose=1, batch_size = 32)
        print(f'test loss: {scores[0]}, test acc:{scores[1]}')

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

        return results

if __name__ == "__main__":
    lstm = LSTM_Embeddings()

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles()

    # link to file https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    fasttext_model = fasttext.load_model('cc.en.300.bin')

    encoder = LabelBinarizer()

    if lstm.args.load_model:
        model = lstm.load_keras_model()
    else:
        # train
        print('Training model')

        # Transform words to fasttext embeddings
        embedded_data = lstm.vectorizer(X_train, fasttext_model)

        # Transform string labels to one-hot encodings
        labels = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back

        # create and train model
        model = lstm.create_model(embedded_data, labels)
        model = lstm.train_model(model, embedded_data, labels)

        # save model 
        lstm.save_keras_model(model)
    
    # run test
    if lstm.args.test and not lstm.args.cop:
        print('Using best estimator on Test set')
        results = lstm.perform_classification(model, X_test, Y_test, fasttext_model, encoder)
    # run dev
    elif not lstm.args.cop:
        print('Using best estimator on Dev set')
        results = lstm.perform_classification(model, X_dev, Y_dev, fasttext_model, encoder)
    
    # test model with COP25 edition
    if lstm.args.cop:
        print(f'Predicting {lstm.args.cop}')
        X_cop, Y_cop = read_single(lstm.args.cop)
        results = lstm.perform_classification(model, X_cop, Y_cop, fasttext_model, encoder)

    lstm.write_run_to_file(vars(lstm.args), results)
