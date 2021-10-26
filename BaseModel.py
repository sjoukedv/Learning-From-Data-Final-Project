from abc import ABC, abstractmethod
import json, argparse, os

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

    # Dev set
    @abstractmethod
    def perform_validation(self):  
        pass

    # Test set
    @abstractmethod
    def perform_classification(self):  
        pass

    def write_run_to_file(self, parameters, results):
        pass
