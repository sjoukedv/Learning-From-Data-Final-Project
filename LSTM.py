#!/usr/bin/env python

'''LSTM fasttext classifier '''

import os
# DEBUG
# fixes cudart64_110.dll error
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import fasttext
import json
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from dataParser import read_articles, read_single
from BaseModel import BaseModel 

class LSTM_Embeddings(BaseModel):
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
        self.name = "LSTM"

    def vectorizer(self, samples, model):
        '''Turn sentence into embeddings, i.e. replace words by the fasttext word vector '''
        return np.array([ model.get_sentence_vector(sample) for sample in samples])

    def create_model(self, X_train, Y_train): 
        model = Sequential()

        learning_rate = 0.01
        loss_function = 'categorical_crossentropy'
        optim = SGD(learning_rate=learning_rate)
        # Take embedding dim and size from emb_matrix
        embedding_dim = len(X_train[0])
        num_tokens = len(X_train)
        num_labels = len(Y_train[0])
        # Now build the model
        model = Sequential()
        model.add(Embedding(num_tokens, embedding_dim,trainable=False))
        # Here you should add LSTM layers (and potentially dropout)
        model.add(LSTM(units = 15))
        
        #raise NotImplementedError("Add LSTM layer(s) here")
        # Ultimately, end with dense layer with softmax
        # model.add(Dense(input_dim=embedding_dim, units=, activation="softmax"))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # Compile model using our settings, check for accuracy
        model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
 
        return model

    def train_model(self, model, X_train, Y_train):
        '''Train the model here. Note the different settings you can experiment with!'''
        # Potentially change these to cmd line args again
        # And yes, don't be afraid to experiment!
        verbose = 1
        epochs = 10 #default 10

        batch_size = 16 #default 32
        
        
        # 10 percent of the training data we use to keep track of our training process
        # Use it to prevent overfitting!
        # validation_split = 0.1
        # , validation_split=validation_split

        # early_acc = EarlyStopping(monitor='accuracy', patience=1)
        # , callbacks=[early_acc]

        # Finally fit the model to our data
        model.fit(X_train, Y_train, verbose=verbose, batch_size=batch_size, epochs=epochs)
        return model

    def write_run_to_file(self, parameters, results):
        res_dir = 'results/' + self.name
        # make sure (sub)directory exists
        os.makedirs(res_dir, exist_ok=True)

        # retrieve version based on number of files in directory
        path, dirs, files = next(os.walk(res_dir))
        version = len(files)

        result = {
            'parameters' : parameters,
            'results' : results
            }

        # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w')) 

    def perform_classification(self, model, X, Y, fasttext_model, encoder):
        test_embedded_data = self.vectorizer(X, fasttext_model)
        # encode gold labels
        true_articles = encoder.fit_transform(Y)

        true_articles = np.argmax(true_articles, axis=1)
        # predict embedded data

        # print(f'Dif array {embedded_data[0] - embedded_data[1]}')
     

        y_pred = model.predict(test_embedded_data, verbose=1)

        Y_pred = np.argmax(Y_pred, axis=1)
        # convert to nearest integer
        # y_pred = np.rint(y_pred, casting='unsafe').astype(int, casting='unsafe')

        print(f'pred labels {type(y_pred)}{y_pred[:50].tolist()}')
        print(f'true labels {type(true_articles)}{true_articles[:50].tolist()}')
        print(classification_report(y_pred, true_articles, labels=encoder.classes_))

if __name__ == "__main__":
    lstm = LSTM_Embeddings()

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles()

    if lstm.args.undersample:
        X_train, Y_train = lstm.under_sample_training_data(X_train, Y_train)

    # link to file https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    fasttext_model = fasttext.load_model('cc.en.300.bin')

    encoder = LabelBinarizer()
    # Transform string labels to one-hot encodings
    labels = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back

    if lstm.args.load_model:
        model = lstm.load_keras_model()
    else:
        # train
        print('Training model')

        # Transform words to fasttext embeddings
        embedded_data = lstm.vectorizer(X_train, fasttext_model)

        # create and train model
        model = lstm.create_model(embedded_data, labels)
      
        model = lstm.train_model(model, embedded_data, labels)
        model.summary()
      
        # save model 
        lstm.save_keras_model(model)
    
    # run test
    if lstm.args.test and not lstm.args.cop:
        print('Using best estimator on Test set')
        results = lstm.perform_classification(model, X_test, Y_test, fasttext_model, encoder)
    # run dev
    elif not lstm.args.cop:
        print('Using best estimator on Dev set')
        results = lstm.perform_classification(model, X_dev, Y_dev, fasttext_model, encoder)
    
    # test model with COP25 edition
    if lstm.args.cop:
        print(f'Predicting {lstm.args.cop}')
        X_cop, Y_cop = read_single(lstm.args.cop)
        results = lstm.perform_classification(model, X_cop, Y_cop, fasttext_model, encoder)

    lstm.write_run_to_file(vars(lstm.args), results)
