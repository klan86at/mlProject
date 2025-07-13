import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_cofig=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBClassifier": KNeighborsRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso()
            }
            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name form dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model on training and testing dataset")

            save_object(
                file_path=self.model_trainer_cofig.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r_square = r2_score(y_test, predicted)

            return r_square
        

        except Exception as e:
            raise CustomException(e, sys)