#!/usr/bin/env python

'''LSTM fasttext classifier '''

import sys
import argparse
import random
import time
import spacy
import fasttext
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate, cross_val_score

from dataParser import read, mergeCopEditions
from BaseModel import BaseModel 

class LSTM(BaseModel):
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

    def split_data(self):
        pass

    def perform_classification(self):
        pass

    def create_model(self):        
        pass

    def perform_cross_validation(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        prepared_data = [ '__label__'+ article['political_orientation'] + ' ' +  article['headline'] for article in articles]

        split = len(prepared_data)//4 * 3
        np.savetxt('fasttext_train.csv', prepared_data[:split], delimiter=',', fmt='%s')
        np.savetxt('fasttext_test.csv', prepared_data[split:], delimiter=',', fmt='%s')

        model = fasttext.train_supervised('fasttext_train.csv')
        print(model)

        # TODO optional GridSearch for values
        return model.test('fasttext_test.csv')

if __name__ == "__main__":
    lstm = LSTM()
    results = lstm.perform_cross_validation()
    lstm.write_run_to_file(vars(lstm.args), results)
