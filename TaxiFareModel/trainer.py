from sklearn import utils
from TaxiFareModel.encoder import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.collect import get_data, clean_data

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
                                       LinearRegression())
        
        return self

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)
    
    def hello(self):
        print('hello')


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
    print(rmse)