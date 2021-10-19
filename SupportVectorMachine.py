#!/usr/bin/env python

'''Support Vector Machine classifier '''

# DEBUG
# fixes cudart64_110.dll error
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

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

class SupportVectorMachine(BaseModel):
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
        self.name = "SupportVectorMachine"
        spacy.prefer_gpu()

        # load spacy
        self.nlp = spacy.load('en_core_web_sm')

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

    def smartJoin(self, x):
        # transform list into string
        return ''.join([i for i in x if i.isalpha()])

    def spacy_pos(self, txt):
        # Part-of-speech transformation
        return [ token.pos_ for token in self.nlp(txt)]    

    def create_model(self):
        count = CountVectorizer(tokenizer=self.spacy_pos)
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
    svm = SupportVectorMachine()
    
    # DEBUG 
    # results = svm.perform_classification()

    results = svm.perform_cross_validation()
    svm.write_run_to_file(vars(svm.args), results)
