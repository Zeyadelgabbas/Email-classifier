import os 
import sys
import pandas as pd
import numpy as np 

from src.exception import CustomException
from src.logger import get_logger 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_preprocessing import DataPreprocessor , DataPreprocessConfig
from model_trainer import ModelTrainer


logging = get_logger(__name__)

## Create paths for training data
@dataclass
class DataIngetstionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","data.csv")
    test_size : int = 0.2


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngetstionConfig()

    def initiate_data_ingestion(self):

        """
        This is where raw data loading is done , splitted into train , test sets and saved into seperate files in artifacts folder
        This method returns the paths of train , test sets
        """
        logging.info("Started data ingestion method")
        try:
            df=pd.read_csv(r'notebook\data\stud.csv')
            logging.info("data set read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            test_size = self.ingestion_config.test_size
            train_set , test_set = train_test_split(df,test_size = test_size , random_state=42)
            logging.info("Train test split done ")
            
            train_set.to_csv(self.ingestion_config.train_data_path,index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index= False, header = True)
            logging.info("train test sets saved to their paths")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e: 
            custom_error = CustomException(e,sys)
            logging.error(custom_error)
            raise custom_error
            

if __name__ == "__main__":


    # st
    obj=DataIngestion()
    train_path , test_path = obj.initiate_data_ingestion()
    
    data_preprocessing = DataPreprocessor()
    X_train , X_val,X_test, y_train , y_val ,y_test  , preprocessor_path = data_preprocessing.initiate_data_preprocessing(train_path=train_path,test_path=test_path)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    print(preprocessor_path)
    model_trainer_obj = ModelTrainer()
    
    print(model_trainer_obj.initiate_model_trainer(X_train,X_val,X_test,y_train,y_val,y_test))
         

                                        