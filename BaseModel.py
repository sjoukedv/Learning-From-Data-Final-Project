import numpy as np
class BaseModel(ABC):
    def __init__(self):
        self.name = "BaseModel"

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
        # results/NaiveBayes/experiment_01.csv
        np.savetxt('results/' + self.name + '/' + 'experiment_' + str(version).zfill(2) + '.csv', results, delimiter=",")
