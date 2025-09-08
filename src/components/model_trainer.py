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
            'decision tree' : DecisionTreeRegressor(),
            'random forest' : RandomForestRegressor(),
            'catboost' : CatBoostRegressor(verbose=False),
            'XGboost' : XGBRegressor(),
            'Kneighbors' : KNeighborsRegressor(),
            'Gradient Boosting':GradientBoostingRegressor(),
            'AdaBoost Regressor':AdaBoostRegressor()
            }

            params={
                "decision tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "random forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "lasso regression":{
                    'alpha' : [0.1,0.5,1,1.5]
                },
                "Ridge regression":{
                    'alpha' : [0.1,0.5,1,1.5]
                },
                "Kneighbors" : {
                    'n_neighbors' : [1 , 3, 5]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear regression":{},
                "XGboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report  , trained_models  = model_evaluate( 
                models,
                X_train , X_val , X_test , 
                y_train , y_val , y_test , params = params 
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
