import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


#todo FIX THE LOAD TRAIN / TEST THINGY
class ModelData:
    def __init__(self, filename: str, sep: str ="|"):
        self.filename = filename
        self.dataset = self.load_dataset_from_csv(sep)


    def load_dataset_from_csv(self,
                              sep: str,
                              ) -> Dataset:
        dataframe = pd.read_csv(self.filename, sep=sep)
        return Dataset.from_pandas(dataframe)

    def get_train_test_splits(self, 
                              X_name :str = "sentence" ,
                              y_name  : str ="appreciative",
                              test_size : float = 0.2):

        return train_test_split( self.dataset[X_name] , self.dataset[y_name], test_size=test_size, shuffle=True) 
    


