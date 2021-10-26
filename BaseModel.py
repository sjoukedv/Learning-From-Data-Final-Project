from abc import ABC, abstractmethod
import json, argparse, os
from joblib import dump, load

class BaseModel(ABC):
    def __init__(self):
        self.name = "BaseModel"
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
        print(f'Storing model to {self.name}.sk.model')
        dump(model, f'models/{self.name}.sk.model')

    # load model to file {self.name}.model.joblib
    def load_sk_model(self):
        print(f'Loading model from {self.name}.sk.model')
        return load(f'models/{self.name}.sk.model')

    def save_keras_model(self, model):
        # TODO
        pass

    def load_keras_model(self):
        # TODO
        pass
