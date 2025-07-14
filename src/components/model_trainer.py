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
                "CatBoosting Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso()
            }
            param_grid = {

            "Decision Tree": {
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['squared_error']
            },

            "Random Forest": {
                'n_estimators': [100],  # Reduced from higher values
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 4],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt'],
                'bootstrap': [True]
            },

            "Linear Regression": {},

            "Gradient Boosting": {
                'n_estimators': [100],
                'learning_rate': [0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'subsample': [0.8]
            },

            "K-Neighbors": {
                'n_neighbors': [3, 5],
                'weights': ['uniform', 'distance'],
                'p': [2]  # Euclidean distance only
            },

            "XGBoost": {
                'n_estimators': [100],
                'learning_rate': [0.1],
                'max_depth': [3, 5],
                'min_child_weight': [1, 3],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'gamma': [0],
                'reg_alpha': [0],
                'reg_lambda': [1],
                'eval_metric': ['rmse']
            },

            "CatBoost": {
                'iterations': [100],
                'learning_rate': [0.1],
                'depth': [4, 6],
                'border_count': [32, 128],
                'l2_leaf_reg': [1, 3],
                'random_strength': [1]
            },

            "AdaBoost": {
                'n_estimators': [50],
                'learning_rate': [0.1, 1.0],
                'loss': ['linear'],
                'base_estimator__max_depth': [2, 3]
            },

            "Ridge": {
                'alpha': [0.1, 1.0],
                'solver': ['auto'],
                'max_iter': [1000],
                'tol': [1e-3]
            },

            "Lasso": {
                'alpha': [0.01, 0.1],
                'selection': ['cyclic'],
                'max_iter': [1000],
                'tol': [1e-3]
            }
        }

        
            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=param_grid
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

            return f"{r_square:.4f}"
        

        except Exception as e:
            raise CustomException(e, sys)