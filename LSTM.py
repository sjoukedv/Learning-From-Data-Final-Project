#!/usr/bin/env python

'''LSTM fasttext classifier '''

import os
# DEBUG
# fixes cudart64_110.dll error
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import json
import numpy as np
import tensorflow as tf
import sys

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TextVectorization
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.initializers import Constant

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
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
    
    def read_embeddings(self, embeddings_file):
        word_embeddings = {}
        f = open(embeddings_file + '.txt', 'r', encoding="utf8") 
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],  dtype='float32')
            word_embeddings[word] = coefs

        f.close()
        return word_embeddings

    def get_emb_matrix(self,voc, emb):
        '''Get embedding matrix given vocab and the embeddings'''
        num_tokens = len(voc) + 2
        word_index = dict(zip(voc, range(len(voc))))
        # Bit hacky, get embedding dimension from the word "the"
        embedding_dim = len(emb["the"])
        # Prepare embedding matrix to the correct size
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = emb.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        # Final matrix with pretrained embeddings that we can feed to embedding layer
        return embedding_matrix

    def create_model(self, Y_train, emb_matrix): 
        learning_rate = 0.01
        loss_function = 'binary_crossentropy'
        # optim = SGD(learning_rate=learning_rate)
        optim = "adam"

        embedding_dim = len(emb_matrix[0])
        num_tokens = len(emb_matrix)
        num_labels = len(Y_train[0])
        # num_labels = 1
        print(f'emd_dim {embedding_dim}, num_tok {num_tokens}, num_lab {num_labels}')

        model = Sequential()
        model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=False))
        model.add(LSTM(3))
        model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="sigmoid"))
        model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
        return model

    def train_model(self, model, X_train, Y_train, X_dev, Y_dev):
        '''Train the model here. Note the different settings you can experiment with!'''
        # Potentially change these to cmd line args again
        # And yes, don't be afraid to experiment!
        verbose = 1
        epochs = 10 #default 10
        batch_size = 32 #default 32

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        # Fit the model to our data
        model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
        self.test_set_predict(model, X_dev, Y_dev, "dev")
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

        # true_articles = np.argmax(true_articles, axis=1)
        # predict embedded data

        # print(f'Dif array {embedded_data[0] - embedded_data[1]}')
     

        y_pred = model.predict(test_embedded_data, verbose=1)

        # Y_pred = np.argmax(Y_pred, axis=1)
        # convert to nearest integer
        # y_pred = np.rint(y_pred, casting='unsafe').astype(int, casting='unsafe')

        print(f'pred labels {type(y_pred)}{y_pred[:50].tolist()}')
        print(f'true labels {type(true_articles)}{true_articles[:50].tolist()}')
        print(classification_report(y_pred, true_articles, labels=encoder.classes_))

    def test_set_predict(self, model, X_test, Y_test, ident):
        '''Do predictions and measure accuracy on our own test set (that we split off train)'''
        # Get predictions using the trained model
        Y_pred = model.predict(X_test)

        print(f'Predictions {Y_pred[:5]}')
        
        # Finally, convert to numerical labels to get scores with sklearn
        # Y_pred = np.argmax(Y_pred, axis=1)

        # print(f'Predictions {Y_pred[:5]}')
        # If you have gold data, you can calculate accuracy
        print(f'Gold labels {Y_test[:5]}')
        Y_test = np.argmax(Y_test, axis=1)
        print(ident)
        print(classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    lstm = LSTM_Embeddings()

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = read_articles()

    if lstm.args.undersample:
        X_train, Y_train = lstm.under_sample_training_data(X_train, Y_train)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    # encoder = LabelEncoder()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    if lstm.args.load_model:
        model = lstm.load_keras_model()
    else:
        # train
        print('Training model')

        # read GloVe word embeddings
        embeddings = lstm.read_embeddings('glove.6B.300d')

        # TextVectorization
        vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
        text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
        vectorizer.adapt(text_ds)
        voc = vectorizer.get_vocabulary()

        emb_matrix = lstm.get_emb_matrix(voc, embeddings)

        # Transform input to vectorized input
        X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
        X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

        # create and train model
        model = lstm.create_model(Y_train_bin, emb_matrix)

        model.summary()

        model = lstm.train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin)
        
        print('saving model')
        # save model 
        lstm.save_keras_model(model)


    sys.exit(0)
    
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