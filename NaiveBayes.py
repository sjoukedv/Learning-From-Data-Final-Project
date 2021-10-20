#!/usr/bin/env python

'''Naive bayes classifier '''

import sys
import argparse
import random
import time
import os
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV

from dataParser import read, mergeCopEditions
from BaseModel import BaseModel 

class NaiveBayes(BaseModel):
    def __init__(self):
        self.arguments = [
        { 
            "command": "-t",
            "refer": "--tfidf",
            "default": None,
            "type": None,
            "action": "store_true",
            "help": "Use the TF-IDF vectorizer instead of CountVectorizer"
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

    def identity(self, x):
        '''Dummy function that just returns the lowercase of the input'''
        return x.lower()

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
        # Use GridSearch to find the best combination of parameters
        return GridSearchCV(
            estimator=Pipeline([('vec', vec), ('cls', MultinomialNB(alpha=1.0, fit_prior=True))]),
            param_grid={
                'cls__alpha': [1.0, 0.9, 0.8, 0.5],
                'cls__fit_prior': [True, False],
            },
            cv=self.args.cv,
            verbose=2
            )


    def perform_cross_validation(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        # extract features
        X_full = [ article['headline'] for article in articles]
        Y_full = [ article['political_orientation'] for article in articles]

        model = self.create_model()
        model.fit(X_full, Y_full)
        print(f'best score {model.best_score_} with params {model.best_params_}')

        return model
   
    def perform_classification(self):
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
            'results' : results.cv_results_,
            'best_score': results.best_score_,
            'best_params': results.best_params_
            }

        for res in results.cv_results_:
            if res == 'params':
                continue
            if hasattr(results.cv_results_[res], "__len__"):
                result['results'][res] = results.cv_results_[res].tolist()

        # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w'))


if __name__ == "__main__":
    nb = NaiveBayes()
    results = nb.perform_cross_validation()
    nb.write_run_to_file(vars(nb.args), results)
