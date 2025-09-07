import os
import sys 
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import get_logger

logging = get_logger(__name__)

def save_object(file_path,obj):
    try: 
        dir_path  = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok = True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

        logging.info(f"file named [{file_path}] saved sucessfully")
    except Exception as e:
        custom_error = CustomException(e,sys)
        logging.error(custom_error)
        raise custom_error
    

def model_evaluate(models,X_train,X_val , X_test , y_train , y_val , y_test):
        
    try :
        model_report = {}
        trained_models={}
        for name , model in models.items():

            model.fit(X_train,y_train)
            pred_vals = model.predict(X_val)
            pred_test = model.predict(X_test)

            val_score = r2_score(y_val,pred_vals)
            test_score = r2_score(y_test,pred_test)

            model_report[name] = (val_score,test_score)
            trained_models[name] = model

        return model_report , trained_models

    except Exception as e:
      custom_error = CustomException(e,sys)
      logging.error(custom_error)
      raise custom_error
     