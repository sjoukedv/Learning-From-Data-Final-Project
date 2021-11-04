#!/usr/bin/env python

''' Fasttext Classifier '''

import os
import json
import numpy as np
import sys

import fasttext

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report

from dataParser import read_articles, read_single
from BaseModel import BaseModel 

class FastText_Model(BaseModel):
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
        self.name = "FastText"

    def format_data(self, X, Y, filename):
        '''Each line of the text file contains a list of labels, followed by the corresponding document. 
        All the labels start by the __label__ prefix, which is how fastText recognize what is a label or what is a word.'''

        new_data = []
        for idx, y in enumerate(Y):
            parsed_sample = '__label__'+ y + ' ' + X[idx]
            new_data.append(parsed_sample)


        f = open(filename, "w", encoding="utf-8")
        f.write('\n'.join(new_data))
        f.close()

    def create_model(self):
        pass
    def train_model(self):
        pass

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

    def perform_classification(self, model, X, Y):
        Y_pred = model.predict(X)[0]
        Y_pred = [ y[0] for y in Y_pred]
        Y_true = [ '__label__'+ y for y in Y] 

        print(classification_report(Y_true, Y_pred, target_names=['left-center', 'right-center'], digits=4))
        return classification_report(Y_true, Y_pred, output_dict=True, target_names=['left-center', 'right-center'])

    def test_set_predict(self, model, X_test, Y_test, ident):
        '''Do predictions and measure accuracy on our own test set (that we split off train)'''
        # Get predictions using the trained model

if __name__ == "__main__":
    ft = FastText_Model()

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles()

    if ft.args.undersample:
        X_train, Y_train = ft.under_sample_training_data(X_train, Y_train)


    if ft.args.load_model:
        model.load_model("models/FastText/fasttext_model.bin")
    else:
        # train
        print('Training model')
        ft.format_data(X_train, Y_train, "fasttext_train_data.txt")
        model = fasttext.train_supervised(input="fasttext_train_data.txt")
        
        print('saving model')
        # save model 
        model.save_model("models/FastText/fasttext_model.bin")

    # run test
    if ft.args.test and not ft.args.cop:
        print('Using best estimator on Test set')
        results = ft.perform_classification(model, X_test, Y_test)
    # run dev
    elif not ft.args.cop:
        print('Using best estimator on Dev set')
        results = ft.perform_classification(model, X_dev, Y_dev)
    
    # test model with COP25 edition
    if ft.args.cop:
        print(f'Predicting {ft.args.cop}')
        X_cop, Y_cop = read_single(ft.args.cop)
        results = ft.perform_classification(model, X_cop, Y_cop)

    ft.write_run_to_file(vars(ft.args), results)
