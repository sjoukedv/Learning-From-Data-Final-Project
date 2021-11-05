from abc import ABC, abstractmethod
import argparse, os
from joblib import dump, load
from tensorflow import keras
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        self.name = "BaseModel"

        # Add argument for under sampling
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

        # Add argument for running on test set
        self.arguments.append(
            { 
                "command": "-test",
                "refer": "--test",
                "default": False,
                "action": "store_true",
                "help": "Run predictions on test set (otherwise uses dev set)"
            }
        )

        # Add argument for loading a model 
        self.arguments.append(
            { 
                "command": "-load",
                "refer": "--load_model",
                "default": False,
                "action": "store_true",
                "help": "Load existing model or perform training"
            }
        )

        # Add argument for using COP
        self.arguments.append(
            {
                "command": "-cop",
                "refer": "--cop",
                "default": None,
                "action": None,
                "type:": str,
                "help": "Path to single COP edition to test (e.g. data/COP25.filt3.sub.json)"
            }
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

    # Store skicit learn model
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

    # Load skicit learn model
    def load_sk_model(self):
        print(f'models/{self.name}/{self.name}.{self.args.model_number}.sk.model')
        return load(f'models/{self.name}/{self.name}.{self.args.model_number}.sk.model')

    # Method used to save a keras model
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

    # Method used to load a keras model
    def load_keras_model(self):
        print(f'models/{self.name}/{self.name}.{self.args.model_number}.keras.model')
        
        model = keras.models.load_model(f'models/{self.name}/{self.name}.{self.args.model_number}.keras.model')
        
        return model
