from TaxiFareModel.encoder import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.collect import get_data, clean_data

import numpy as np

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

import joblib

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FRA [Paris] [fabienpardo] linear 2" 

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = make_pipeline(DistanceTransformer(),
                                   StandardScaler())
        
        time_pipe = make_pipeline(TimeFeaturesEncoder('pickup_datetime'),
                                   OneHotEncoder(handle_unknown='ignore'))
        
        preproc_pipe = make_column_transformer((dist_pipe,["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                                               (time_pipe, ['pickup_datetime']),
                                               remainder='drop')
        
        self.pipeline = make_pipeline(preproc_pipe,
                                       GradientBoostingRegressor(alpha=0.6801648976143877, learning_rate=0.07759030694642681))
        
        return self
    
    def search(self):
        
        rmse = make_scorer(mean_squared_error, squared=False)

        grid = {'gradientboostingregressor__alpha': np.random.uniform(.001,1,10),
                'gradientboostingregressor__learning_rate': np.random.uniform(.001,1,10)}


        search = GridSearchCV(self.pipeline, grid, 
                                    scoring=rmse,
                                    cv=5,
                                    n_jobs=-1)
        
        search.fit(self.X, self.y)
        return search.best_score_, search.best_params_
    
    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)
    
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'TaxiFarelinear.joblib')
        return self
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data()
    df =  clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    train = Trainer(X_train,y_train)
    train.set_pipeline()
    train.run()
    rmse = train.evaluate(X_val, y_val)

    train.mlflow_log_param('model', 'linear')
    train.mlflow_log_metric('rmse', rmse)
    
    train.save_model()
    
    print(rmse)