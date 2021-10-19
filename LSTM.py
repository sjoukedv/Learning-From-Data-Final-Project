#!/usr/bin/env python

'''LSTM classifier '''

import sys
import argparse
import random
import time
import spacy

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


    # TODO remove because built-in functionality of cross_validate
    def split_data(self, X_full, Y_full, test_percentage):
        ## This method is responsible for splitting the data into test and training sets, based on the percentage. 
        ## The two training and two test sets are returned. 
        split_point = int((1.0 - test_percentage)*len(X_full))

        X_train = X_full[:split_point]
        Y_train = Y_full[:split_point]
        X_test = X_full[split_point:]
        Y_test = Y_full[split_point:]
        return X_train, Y_train, X_test, Y_test


    def identity(self, x):
        '''Dummy function that just returns the input'''
        return x

    def create_model(self):
        count = CountVectorizer(preprocessor=self.smartJoin, tokenizer=self.spacy_pos)
        tf_idf = TfidfVectorizer(preprocessor=self.identity, tokenizer=self.identity,max_df=0.8, min_df=0.0001, ngram_range=(1,3))
        union = FeatureUnion([("tf_idf", tf_idf),("count", count)])
        
        # Combine the union feature with a LinearSVC
        return Pipeline([("union", union),('cls', LinearSVC(C=10))])

    def perform_cross_validation(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        # extract features
        X_full = [ article['body'] for article in articles]
        Y_full = [ article['political_orientation'] for article in articles]

        model = self.create_model()

        # TODO optional GridSearch for value of e.g. C
        return cross_validate(model, X_full, Y_full, cv=3, verbose=1)

if __name__ == "__main__":
    lstm = LSTM()
    results = lstm.perform_cross_validation()
    lstm.write_run_to_file(vars(lstm.args), results)
