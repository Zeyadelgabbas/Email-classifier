import sys 
import os 
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder 

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object


logging = get_logger(__name__)

@dataclass
class DataPreprocessConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataPreprocessor:
    def __init__(self):
        self.data_preprocess_config = DataPreprocessConfig()

    def simple_data_preprocessor(self,num_features,cat_features):

        """
        Simple pipeline for handling numerical and categorical features

        """
        try:
            num_pipeline = Pipeline(
                [
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler()),
                ]
            )
            
            cat_pipeline = Pipeline(
                [
                    ("cat_imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder())

                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
            logging.info("data preprocessor created sucessfully ")
            return preprocessor

        except Exception as e :
            custom_error = CustomException(e,sys)
            logging.error(custom_error)
            raise custom_error

    def initiate_data_preprocessing(self,train_path,test_path):

        """
        
        """

        try : 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("reading train and test data completed ")
            logging.info("obtaining preprocessor object")

            
            
            target_column_name = "math_score"

            train_df_input = train_df.drop(columns = target_column_name,axis = 1)
            train_df_target = train_df[target_column_name]

            test_df_input = test_df.drop(columns = target_column_name,axis = 1)
            test_df_target = test_df[target_column_name]

            num_features = train_df_input.select_dtypes(include=["float64","int64","int32","float32"]).columns.to_list()
            cat_features = train_df_input.select_dtypes(include=["object"]).columns.to_list()

            preprocessor = self.simple_data_preprocessor(num_features=num_features,cat_features=cat_features)

            logging.info("applying preprocessing on input features")

            train_df_input , val_df_input , train_df_target , val_df_target = train_test_split(
                            train_df_input,train_df_target,test_size=0.2,random_state=42

            )
            

            train_input_arr = preprocessor.fit_transform(train_df_input)
            test_input_arr = preprocessor.transform(test_df_input)
            val_input_arr = preprocessor.transform(val_df_input)

            logging.info("preprocessing completed ")

            save_object(
                self.data_preprocess_config.preprocessor_obj_file_path,
                preprocessor
            )

            return(
                train_input_arr ,val_input_arr, test_input_arr ,
                train_df_target , val_df_target,test_df_target ,
                self.data_preprocess_config.preprocessor_obj_file_path     # path to my preprocessor object 
            )


        except Exception as e:
            custom_error = CustomException(e,sys)
            logging.error(custom_error)
            raise custom_error
            



