'''Support Vector Machine classifier'''

# DEBUG
# fixes cudart64_110.dll error
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import os
import json
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

from dataParser import read_articles, read_single
from BaseModel import BaseModel 
from sklearn.model_selection import GridSearchCV

class SupportVectorMachine(BaseModel):
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
        self.name = "SupportVectorMachine"
        spacy.prefer_gpu()

        # load spacy
        self.nlp = spacy.load('en_core_web_sm')

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

        return GridSearchCV(
            # Combine the union feature with a LinearSVC
            estimator=Pipeline([("union", union),('cls', LinearSVC())]),
            param_grid=self.param_grid,
            cv=self.args.cv,
            verbose=3
        )

    def train_model(self, model, X_train, Y_train):
        model = model.fit(X_train, Y_train)
      
        self.gs_cv_results = model.cv_results_
        self.gs_best_params = model.best_params_
        self.gs_best_score = model.best_score_
        print(f'best training score {model.best_score_} with params {model.best_params_}')
   
        # return best estimator
        return model.best_estimator_
    
    # Normal classification for external test sets
    def perform_classification(self, model, X, Y):
        Y_pred = model.predict(X)
        print(classification_report(Y, Y_pred, target_names=['left-center', 'right-center'], digits=4))
        return classification_report(Y, Y_pred, output_dict=True, target_names=['left-center', 'right-center'], digits=3)

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
    svm = SupportVectorMachine()

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles() 

    if svm.args.undersample:
        X_train, Y_train = svm.under_sample_training_data(X_train, Y_train)

    svm.param_grid = {
            'union__tf_idf__max_df': [0.5],
            'union__tf_idf__min_df': [0.0001],
            'union__tf_idf__ngram_range': [(1,3)],
            'cls__C': [0.5]
        }

    if svm.args.load_model:
        model = svm.load_sk_model()
    else:
        # train
        print('Training model')
        model = svm.create_model()
        model = svm.train_model(model, X_train, Y_train)
    
        # save model
        svm.save_sk_model(model)

    # run test
    if svm.args.test and not svm.args.cop:
        print('Using best estimator on Test set')
        results = svm.perform_classification(model, X_test, Y_test)
    # run dev
    elif not svm.args.cop:
        print('Using best estimator on Dev set')
        results = svm.perform_classification(model, X_dev, Y_dev)

    # test model with COP25 edition
    if svm.args.cop:
        print(f'Predicting {svm.args.cop}')
        X_cop, Y_cop = read_single(svm.args.cop)
        results = svm.perform_classification(model, X_cop, Y_cop)
        
    svm.write_run_to_file(vars(svm.args), results)
