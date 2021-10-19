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


    @abstractmethod
    def create_model(self):
        pass 
    
    @abstractmethod
    def train_model(self):
        pass 
    
    @abstractmethod
    def test_model(self):
        pass 
    
    @abstractmethod
    def evaluate_model(self):
        pass 
    
    @abstractmethod
    def perform_classification(self):
        pass

    @abstractmethod
    def split_data(self):
        pass

    @abstractmethod
    def perform_classification(self):
        pass 

    def write_run_to_file(self, version, parameters, results):
        # make sure (sub)directory exists
        os.makedirs('results/' + self.name, exist_ok=True)

        # write results to file
        json.dump({
            'parameters' : parameters,
            'results' : results
            }, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w'))
