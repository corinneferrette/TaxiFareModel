from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import pandas as pd
from TaxiFareModel.data import get_data

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

        X.key = pd.to_datetime(X[self.time_column])
        X.key = X.key.tz_convert(self.time_zone_name)
        X["dow"] = X.key.day_of_week
        X["hour"] = X.key.hour
        X["month"] = X.key.month
        X["year"] = X.key.year
        return X[[
                'dow','hour','month','year'
            ]].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    pass


if __name__ == '__main__':
    encoder=TimeFeaturesEncoder()
    data=get_data()
    print(data)
