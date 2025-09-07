import os 
import sys 
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object
from src.utils import model_evaluate
logging = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_train ,X_val, X_test , y_train ,y_val, y_test ):

        try:
            models = {
            'linear regression' : LinearRegression(),
            'lasso regression' : Lasso(),
            'Ridge regression' : Ridge(),
            'SVM regressor' : SVR(),
            'decision tree' : DecisionTreeRegressor(),
            'random forest' : RandomForestRegressor(),
            'catboost' : CatBoostRegressor(verbose=False),
            'XGboost' : XGBRegressor(),
            'Kneighbors' : KNeighborsRegressor(),
            }
            model_report  , trained_models  = model_evaluate( 
                models,
                X_train , X_val , X_test , 
                y_train , y_val , y_test 
            )

            best_model_name = max(model_report , key = lambda k: model_report[k][0])
            best_model_score = model_report[best_model_name]

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = trained_models[best_model_name]
            )

            return(
                best_model_score
            )

        except Exception as e:
            custom_error = CustomException(e,sys)
            logging.error(custom_error)
            raise custom_error
