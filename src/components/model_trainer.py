import sys 
import os
from dataclasses import dataclass

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path: str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_moldel_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and Testing Dataset")

            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "LinearRegression":LinearRegression(),
                "SupportVectorRegressor":SVR(),
                "XGBRegressor":XGBRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "CategoricalBoostingRegressor":CatBoostRegressor(),
                "KneighborsRegressor":KNeighborsRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "AdaptiveBoostRegressor":AdaBoostRegressor(),
                "RandomForestRegressor":RandomForestRegressor()
            }

            model_report: dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            # To get best model Score
            best_model_score=max(sorted(model_report.values()))

            # To get best model Name
            best_model_name = max(model_report, key=model_report.get)

            # Extract best model name from above models Dictionary
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model Found !!!")
            
            logging.info("Best model found on Both Training and Testing DataSet")

            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)