'''Support Vector Machine classifier'''

# DEBUG
# fixes cudart64_110.dll error
import os
import json
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

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
from sklearn.model_selection import GridSearchCV

class SupportVectorMachine(BaseModel):
    def __init__(self):
        self.arguments = [
            { 
                "command": "-ts",
                "refer": "--test_file",
                "default": None,
                "action": None,
                "type": str,
                "help": "Test file to run predictions on (e.g. COP24.filt3.sub.json)"
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
                "command": "-C",
                "refer": "--C",
                "default": 0.1,
                "action": None,
                "type": float,
                "help": "Regularization parameter"
            },
            {
                "command": "-min_df",
                "refer": "--min_df",
                "default": 0.0001,
                "action": None,
                "type": float,
                "help": "Minimum document frequency"
            },
            {
                "command": "-max_df",
                "refer": "--max_df",
                "default": 0.8,
                "action": None,
                "type": float,
                "help": "Maximum document frequency"
            },
            {
                "command": "-ngram_range",
                "refer": "--ngram_range",
                "default": (1,3),
                "action": None,
                "type": tuple,
                "help": "The lower and upper boundary of the range of n-value for different n-grams"
            },
        ]

        super().__init__()
        self.name = "SupportVectorMachine"
        spacy.prefer_gpu()

        # load spacy
        self.nlp = spacy.load('en_core_web_sm')

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
            'best_params': results.best_params_,
            'param_grid': self.param_grid,
            }

        for res in results.cv_results_:
            if res == 'params':
                continue
            if hasattr(results.cv_results_[res], "__len__"):
                result['results'][res] = results.cv_results_[res].tolist()

        # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w'))

    def identity(self, x):
        '''Dummy function that just returns the lowercase of the input'''
        return x.lower()

    def smartJoin(self, x):
        # transform list into string
        return ''.join([i.lower() for i in x if i.isalpha()])

    def spacy_pos(self, txt):
        # Part-of-speech transformation
        return [ token.pos_.lower() for token in self.nlp(txt.lower())]    

    # Create the model using gridsearch
    def create_model(self):
        count = CountVectorizer(preprocessor=self.smartJoin, tokenizer=self.spacy_pos)
        tf_idf = TfidfVectorizer(preprocessor=self.identity, tokenizer=self.identity)
        union = FeatureUnion([("tf_idf", tf_idf),("count", count)])

        # Initialise parameters
        self.param_grid = {
            'union__tf_idf__max_df': [1.0, 0.75, 0.5],
            'union__tf_idf__min_df': [0.0001, 0.001, 0.01],
            'union__tf_idf__ngram_range': [(1,3), (2,3), (3,3)],
            'cls__C': [0.1, 0.5, 0.05]
        }

        return GridSearchCV(
            # Combine the union feature with a LinearSVC
            estimator=Pipeline([("union", union),('cls', LinearSVC())]),
            param_grid=self.param_grid,
            cv=self.args.cv,
            verbose=3
        )

    def final_model(self):
        count = CountVectorizer(preprocessor=self.smartJoin, tokenizer=self.spacy_pos)
        tf_idf = TfidfVectorizer(preprocessor=self.identity, tokenizer=self.identity, max_df=1.0, min_df=0.0001, ngram_range=(2,3))
        union = FeatureUnion([("tf_idf", tf_idf),("count", count)])

        return Pipeline([("union", union),('cls', LinearSVC(C=5))])
        
    # Performs cross validation using gridsearch via create_model
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
    
    # Normal classification for external test sets
    def perform_classification(self):
        # The documents and labels are retrieved. 
        data = read()
        articles = mergeCopEditions(data)

        # extract features
        Y_full = [ article['political_orientation'] for article in articles]
        X_full = [ article['body'] for article in articles]

        model = self.final_model()
        model.fit(X_full, Y_full)

        # read test data
        test_articles = read_single(self.args.test_file)

        # extract headlines
        test_parsed_articles = [ article['headline'] for article in test_articles]

        pred_articles = model.predict(test_parsed_articles)
        true_articles = [ article['political_orientation'] for article in test_articles]

        print(classification_report(true_articles, pred_articles))

if __name__ == "__main__":
    svm = SupportVectorMachine()
    
    if svm.args.test_file:
        svm.perform_classification
    else:
        results = svm.perform_cross_validation()
        svm.write_run_to_file(vars(svm.args), results)
