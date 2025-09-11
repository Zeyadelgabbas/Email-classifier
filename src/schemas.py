from pydantic import BaseModel ,Field
from typing import Annotated , Literal


class Prediction_features(BaseModel):

#gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score

    gender : Annotated[Literal['male','female'] ,Field(...,description="gender") ]
    race_ethnicity : Annotated[Literal['group A','group B','group C','group D','group E'] , Field('group A')]
    parental_level_of_education: str = Field(..., alias="parental_education")
    lunch : str 
    test_preparation_course : Annotated[Literal['none','completed'],Field('none')]
    reading_score : Annotated[float, Field(...,ge = 0 , lt = 100)]
    writing_score : Annotated[float, Field(...,ge = 0 , lt = 100)]



