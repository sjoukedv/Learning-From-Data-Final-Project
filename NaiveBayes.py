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

from dataParser import read, read_single, mergeCopEditions
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
        },
            { 
            "command": "-i",
            "refer": "--test_file",
            "default": None,
            "action": None,
            "type": str,
            "help": "Test file to run predictions on (default COP24.filt3.sub.json)"
        },
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
                'cls__alpha': [1.0, 0.75, 0.5],
                # 'cls__fit_prior': [True, False],
                # 'vec__ngram_range' : [(1,1), (1,2), (1,3), (2,3)],
                # 'vec__analyzer': ['word', 'char', 'char_wb'],
                # 'vec__max_df': [1.0, 0.9, 0.8],
                # 'vec__min_df': [1, 0.9, 0.8],
                'vec__max_features': [4,8,16, None],
            },
            cv=self.args.cv,
            verbose=2
            )
    
    def final_model(self):
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
        return Pipeline([('vec', vec), ('cls', MultinomialNB(alpha=1.0, fit_prior=True))])

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
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        # extract features
        X_full = [ article['headline'] for article in articles]
        Y_full = [ article['political_orientation'] for article in articles]

        model = self.final_model()
        model.fit(X_full, Y_full)

        # read test data
        test_articles = read_single(self.args.test_file)
        # extract headlines
        test_parsed_articles = [ article['headline'] for article in test_articles]

        pred_articles = model.predict(test_parsed_articles)
        true_articles =  [ article['political_orientation'] for article in test_articles]
    
        print(classification_report(true_articles, pred_articles))

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

    # run test
    if nb.args.test_file:
        nb.perform_classification()
    # run dev
    else:
        results = nb.perform_cross_validation()
        nb.write_run_to_file(vars(nb.args), results)
