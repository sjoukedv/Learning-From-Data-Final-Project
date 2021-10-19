#!/usr/bin/env python

'''Naive bayes classifier '''

import sys
import argparse
import random
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate, cross_val_score

from dataParser import read, mergeCopEditions
from BaseModel import BaseModel 

class NaiveBayes(BaseModel):
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
        },
        {
            "command": "-cv",
            "refer": "--cv",
            "default": 3,
            "action": None,
            "type:": int,
            "help": "Determines the cross-validation splitting strategy"
        }
        ]

        super().__init__()
        self.name = "NaiveBayes"

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
        # Convert the texts to vectors
        # We use a dummy function as tokenizer and preprocessor,
        # since the texts are already preprocessed and tokenized.
        if self.args.tfidf:
            vec = TfidfVectorizer(preprocessor=self.identity, tokenizer=self.identity)
        else:
            # Bag of Words vectorizer
            vec = CountVectorizer(preprocessor=self.identity, tokenizer=self.identity)

        # Combine the vectorizer with a Naive Bayes classifier
        return Pipeline([('vec', vec), ('cls', MultinomialNB(alpha=1.0, fit_prior=True))])

    def perform_cross_validation(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        # extract features
        X_full = [ article['body'] for article in articles]
        Y_full = [ article['political_orientation'] for article in articles]

        model = self.create_model()

        # TODO optional GridSearch for value of e.g. alpha
        return cross_validate(model, X_full, Y_full, cv=self.args.cv, verbose=1)
   
    
    # TODO remove because cross_validate does this for us
    def perform_classification(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        # extract features
        Y_full = [ article['political_orientation'] for article in articles]
        X_full = [ article['body'] for article in articles]

        # The documents and labels are split into a training and test set. 
        X_train, Y_train, X_test, Y_test = self.split_data(X_full, Y_full, 0.3)

        model = self.create_model()

        # DEBUG
        # t0 = time.time()

        model = model.fit(X_train, Y_train)

        # DEBUG
        # print("Training time: ", time.time() - t0)

        Y_pred = model.predict(X_test)

        return classification_report(Y_test, Y_pred, digits=3, output_dict=True)

if __name__ == "__main__":
    nb = NaiveBayes()
    
    # DEBUG 
    # results = nb.perform_classification()

    results = nb.perform_cross_validation()
    nb.write_run_to_file(vars(nb.args), results)
