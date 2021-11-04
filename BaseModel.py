from abc import ABC, abstractmethod
import argparse, os
from joblib import dump, load
from tensorflow import keras
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        self.name = "BaseModel"

        # Add argument for down sampling
        self.arguments.append(
            {
                "command": "-undersample",
                "refer": "--undersample",
                "default": None,
                "type": None,
                "action": "store_true",
                "help": "Value which indicates whether to downsample the data"
            },
        )
        
        # Add argument for model number to load
        self.arguments.append(
            {
            "command": "-model_number",
            "refer": "--model_number",
            "default": '00',
            "action": None,
            "type:": str,
            "help": "Name of model which should be loaded"
            },
        )

        self.args = self.create_arg_parser()

    def create_arg_parser(self):
        parser = argparse.ArgumentParser()

        for argument in self.arguments:

            if argument.get("type") != None:
                parser.add_argument(
                    argument["command"], 
                    argument["refer"],
                    default=argument.get("default"),
                    type=argument.get("type"),
                    help=argument.get("help"),
                    action=argument.get("action")
                )
            else:
                parser.add_argument(
                    argument["command"], 
                    argument["refer"],
                    default=argument.get("default"),
                    help=argument.get("help"),
                    action=argument.get("action")
                )
        
        args = parser.parse_args()
        return args

    # Method used to under sample training data
    def under_sample_training_data(self, X_train, Y_train):
        # define undersample strategy
        undersample = RandomUnderSampler(sampling_strategy='majority')
        # apply undersampling
        X_train, Y_train = undersample.fit_resample(np.array(X_train).reshape(-1, 1), np.array(Y_train).reshape(-1, 1))
        return X_train.flatten().tolist(), Y_train.flatten().tolist()

    # Create model that can be fitted to the train data
    @abstractmethod
    def create_model(self):
        pass 

    # Train set
    @abstractmethod
    def train_model(self):
        pass 

    # Dev or Test set
    @abstractmethod
    def perform_classification(self):  
        pass

    def write_run_to_file(self, parameters, results):
        pass

    # store model to file {self.name}.model.joblib
    def save_sk_model(self, model):
        res_dir = 'models/' + self.name
        # make sure (sub)directory exists
        os.makedirs(res_dir, exist_ok=True)

        # retrieve version based on number of files in directory
        path, dirs, files = next(os.walk(res_dir))
        version = len(files)

        store_location = f'models/{self.name}/{self.name}.{str(version).zfill(2)}.sk.model'

        print(f'Storing model to {store_location}')
        dump(model, store_location)

    # load model to file {self.name}.model.joblib
    def load_sk_model(self):
        print(f'models/{self.name}/{self.name}.{self.args.model_number}.sk.model')
        return load(f'models/{self.name}/{self.name}.{self.args.model_number}.sk.model')

    def save_keras_model(self, model):
        res_dir = 'models/' + self.name
        # make sure (sub)directory exists
        os.makedirs(res_dir, exist_ok=True)

        # retrieve version based on number of files in directory
        path, dirs, files = next(os.walk(res_dir))
        version = len(files)

        store_location = f'models/{self.name}/{self.name}.{str(version).zfill(2)}.keras.model'

        print(f'Storing model to {store_location}')
        model.save(store_location)

    def load_keras_model(self):
        print(f'models/{self.name}/{self.name}.{self.args.model_number}.keras.model')
        
        model = keras.models.load_model(f'models/{self.name}/{self.name}.{self.args.model_number}.keras.model')
        
        #if self.name == 'Bert':
            #model.load_weights(f'models/{self.name}/{self.name}.{self.args.model_number}.h5')
        
        return model
