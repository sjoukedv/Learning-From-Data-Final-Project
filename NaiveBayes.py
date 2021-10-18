#!/usr/bin/env python

'''Naive bayes classifier '''

import sys
import argparse
import random
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from sklearn.metrics import precision_score # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
from sklearn.metrics import recall_score # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
from sklearn.metrics import f1_score # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
from sklearn.metrics import confusion_matrix # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score

from dataParser import read, mergeCopEditions
from sklearn.metrics import classification_report
from BaseModel import BaseModel 

class NaiveBayes(BaseModel):
    def __init__(self):
        self.name = "NaiveBayes"
        self.args = self.create_arg_parser()

    def create_arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
                            help="Input file to learn from (default reviews.txt)")
        parser.add_argument("-t", "--tfidf", action="store_true",
                            help="Use the TF-IDF vectorizer instead of CountVectorizer")
        parser.add_argument("-tp", "--test_percentage", default=0.20, type=float,
                            help="Percentage of the data that is used for the test set (default 0.20)")
        args = parser.parse_args()
        return args

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

    def perform_cross_validation(self, use_sentiment, input_file):
        data = read()
        #articles, orientations = mergeCopEditions(data)
        # The documents and labels are retrieved. 
        X_full, Y_full = mergeCopEditions(data)

        # Convert the texts to vectors
        # We use a dummy function as tokenizer and preprocessor,
        # since the texts are already preprocessed and tokenized.
        if self.args.tfidf:
            vec = TfidfVectorizer(preprocessor=self.identity, tokenizer=self.identity)
        else:
            # Bag of Words vectorizer
            vec = CountVectorizer(preprocessor=self.identity, tokenizer=self.identity)

        # Combine the vectorizer with a Naive Bayes classifier
        classifier = Pipeline([('vec', vec), ('cls', MultinomialNB(alpha=1.0, fit_prior=True))])

        print("Cross validation test scores: {}".format(cross_validate(classifier, X_full, Y_full, cv=5)['test_score']))

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
        classifier = Pipeline([('vec', vec), ('cls', MultinomialNB(alpha=1.0, fit_prior=True))])

        return classifier

    def train_model(self, model, X_train, Y_train):
        # The must be fitted to the model. Meaning that it must learn from the model. Done by passing the training set to the fit method. 
        return model.fit(X_train, Y_train)

    def test_model(self, model, X_test):
        # The fitted classifier is used to do a prediction based on the test documents. 
        return model.predict(X_test)

    def evaluate_model(self, Y_pred, Y_test):
        results = classification_report(Y_test, Y_pred, digits=3, output_dict=True)
        print(results)
        return results

    def perform_classification(self):
        # The documents and labels are retrieved. 
        data = read()
        X_full, Y_full = mergeCopEditions(data)

        # The documents and labels are split into a training and test set. 
        X_train, Y_train, X_test, Y_test = self.split_data(X_full, Y_full, 0.3)

        model = self.create_model()

        t0 = time.time()

        model = self.train_model(model, X_train, Y_train)

        print("Training time: ", time.time() - t0)

        Y_pred = self.test_model(model, X_test)

        self.results = self.evaluate_model(Y_pred, Y_test)

if __name__ == "__main__":
    model = NaiveBayes()
    model.perform_classification()
    model.write_run_to_file("0", [], [])
    #args = create_arg_parser()

    #print("----Six-class classification----")
    #perform_classification(False, args.input_file)
    #print("----Cross-validate results----")
    #perform_cross_validation(False, args.input_file)
    
