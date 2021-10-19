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
    def split_data(self):
        pass

    @abstractmethod
    def create_model(self):
        pass 

    @abstractmethod
    def perform_classification(self):
        pass

    @abstractmethod
    def perform_cross_validation(self):
        pass 

    def write_run_to_file(self, version, parameters, results):
        # make sure (sub)directory exists
        os.makedirs('results/' + self.name, exist_ok=True)

        result = {
            'parameters' : parameters,
            'results' : results
            }

        # convert array to list
        # TODO comment this for loop when using single classification
        for res in results:
            if hasattr(results[res], "__len__"):
                result['results'][res] = results[res].tolist()

        # write results to file
        json.dump(result, open('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.json', 'w'))
