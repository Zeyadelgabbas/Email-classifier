import sys
import os
from src.utils import load_object 
from src.exception import CustomException 
from src.logger import get_logger
import pandas as pd 

logging = get_logger(__name__)





class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            features = preprocessor.transform(features)
            preds  = model.predict(features)
            logging.info(f"prediction completed : {preds[0]}")
            return preds[0]
        
        except Exception as e : 
            custom_error = CustomException(e,sys)
            logging.error(custom_error)
            raise custom_error
        

class CustomData:

    def __init__(self):
        pass
    def data_preparation(self,dictionary_features):
        try:    
            df = pd.DataFrame([dictionary_features])
            logging.info("data prepared successfully ")
            return df 
        except Exception as e : 
            custom_error = CustomException(e,sys)
            logging.error(custom_error)
            raise custom_error
    """
    def __init__(self, gender , race_ethnicity , parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch 
        self.test_preparation_course  = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

"""