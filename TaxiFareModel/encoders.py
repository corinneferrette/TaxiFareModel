from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import pandas as pd
import numpy as np
from TaxiFareModel.data import get_data
from TaxiFareModel.utils import haversine_vectorized



class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column='key', time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X.index = pd.to_datetime(X[self.time_column])
        # X.index = X.index.tz_convert(self.time_zone_name)
        X["dow"] = X.index.day_of_week
        X["hour"] = X.index.hour
        X["month"] = X.index.month
        X["year"] = X.index.year
        return X[['dow', 'hour', 'month', 'year']].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self,
                 start_lat='pickup_latitude',
                 start_lon='pickup_longitude',
                 end_lat='dropoff_latitude',
                 end_lon='dropoff_longitude'):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X['haversine_dist'] = haversine_vectorized(X,
                                                   start_lat=self.start_lat,
                                                   start_lon=self.start_lon,
                                                   end_lat=self.end_lat,
                                                   end_lon=self.end_lon)
        return X[['haversine_dist']]


if __name__ == '__main__':
    # encoder=TimeFeaturesEncoder()
    distance=DistanceTransformer()
    data=get_data()
    # print(encoder.transform(data))
    print(distance.transform(data))
