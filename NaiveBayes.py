#!/usr/bin/env python

'''Naive bayes classifier '''

import os
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from dataParser import read_articles, read_single
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
        self.name = "NaiveBayes"

    def identity(self, x):
        return x.lower()

    def create_model(self):
        if self.args.tfidf:
            vec = TfidfVectorizer(preprocessor=self.identity, tokenizer=self.identity)
        else:
            vec = CountVectorizer(preprocessor=self.identity, tokenizer=self.identity)

        # Combine the vectorizer with a Naive Bayes classifier
        # Use GridSearch to find the best combination of parameters
        return GridSearchCV(
            estimator=Pipeline([('vec', vec), ('cls', MultinomialNB(alpha=0.75, fit_prior=True))]),
            param_grid=self.param_grid,
            cv=self.args.cv,
            verbose=2
            )
    
    def train_model(self, model, X_train, Y_train):
        model = model.fit(X_train, Y_train)
      
        self.gs_cv_results = model.cv_results_
        self.gs_best_params = model.best_params_
        self.gs_best_score = model.best_score_
        print(f'best training score {model.best_score_} with params {model.best_params_}')
   
        # return best estimator
        return model.best_estimator_

    def perform_classification(self, model, X, Y):
        Y_pred = model.predict(X)
        print(classification_report(Y, Y_pred, target_names=['left-center', 'right-center']))
        return classification_report(Y, Y_pred, output_dict=True, target_names=['left-center', 'right-center'])
 
    def write_run_to_file(self, parameters, results):
        res_dir = 'results/' + self.name
        # make sure (sub)directory exists
        os.makedirs(res_dir, exist_ok=True)

        # retrieve version based on number of files in directory
        path, dirs, files = next(os.walk(res_dir))
        version = len(files)

        result = {
            'parameters' : parameters,
            'param_grid': self.param_grid,
            'classification_report': results
            }

        # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w'))

if __name__ == "__main__":
    nb = NaiveBayes()

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles() 
    X_train, Y_train = nb.under_sample_training_data(X_train, Y_train)

    nb.param_grid = {
        'cls__alpha': [1.0, 0.75, 0.5],
        # 'cls__fit_prior': [True, False],
        # 'vec__ngram_range' : [(1,1), (1,2), (1,3), (2,3)],
        # 'vec__analyzer': ['word', 'char', 'char_wb'],
        # 'vec__max_df': [1.0, 0.9, 0.8],
        # 'vec__min_df': [1, 0.9, 0.8],
        # 'vec__max_features': [4,8, None],
    }

    if nb.args.load_model:
        model = nb.load_sk_model()
    else:
        # train
        print('Training model')
        model = nb.create_model()
        model = nb.train_model(model, X_train, Y_train)
    
        # save model
        nb.save_sk_model(model)

    # run test
    if nb.args.test and not nb.args.cop:
        print('Using best estimator on Test set')
        results = nb.perform_classification(model, X_test, Y_test)
    # run dev
    elif not nb.args.cop:
        print('Using best estimator on Dev set')
        results = nb.perform_classification(model, X_dev, Y_dev)

    # test model with COP25 edition
    if nb.args.cop:
        print(f'Predicting {nb.args.cop}')
        X_cop, Y_cop = read_single(nb.args.cop)
        results = nb.perform_classification(model, X_cop, Y_cop)
        
    nb.write_run_to_file(vars(nb.args), results)
