from abc import ABC, abstractmethod
import pandas 

class BaseModel(ABC):
    def __init__(self):
        self.name = "Base model"

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
        filename = "./nb_experiments/run_" + self.name + "_" + version

        file1 = open(filename + ".txt","w")
  
        # \n is placed to indicate EOL (End of Line)
        file1.write(self.name + ": " + "Experiment " + version + "\n")

        for arg in vars(self.args):
            file1.write(str(arg) + ": " + str(getattr(self.args, arg)) + "\n")

        file1.close() #to change file access modes

        df = pandas.DataFrame(self.results).transpose()
        df.to_csv(filename + ".csv")




    


    