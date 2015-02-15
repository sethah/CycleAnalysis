import numpy as np
import pandas as pd
import pymongo
from SignalProc import weighted_average, smooth, diff
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from StravaEffort import StravaActivity
import matplotlib.pyplot as plt
from PlotTools import PlotTool
import seaborn as sns

"""Need to rethink the filtering function"""

# global vars
meters_per_mile = 1609.34
feet_per_meter = 3.280


class StravaUser(object):

    def __init__(self, name):
        self.activities = None
        self.name = name
        self.get_activities()

    def get_activities(self, query={}, min_length=990):
        client = pymongo.MongoClient()
        db = client.mydb
        table = db.activities

        result = table.find(query)
        activities = [activity for activity in result]

        self.activities = []
        for activity in activities:
            a = StravaActivity(activity)
            if a.is_ride() and len(a.distance.raw_data) > min_length:
                self.activities.append(a)

    def make_df(self, activities=None):
        if activities is not None:
            start = activities[0]
            stop = activities[1]
        else:
            start = 0
            stop = len(self.activities)

        df = None
        for a in self.activities[start:stop]:
            if df is None:
                df = a.hills_df()
            else:
                df = df.append(a.hills_df(), ignore_index=True)

        return df


if __name__ == '__main__':
    u = StravaUser('Seth')
    df = u.make_df()
    df = df[df['velocity'] > 3]
    y = df.pop('velocity')
    # df.pop('altitude')
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    # rf.score(X_test, y_test)
    cvscore = cross_val_score(rf, X_train, y_train, cv=10, scoring='r2')
    print np.mean(cvscore)
    importances = sorted(zip(rf.feature_importances_, df.columns), key=lambda x: x[0])
    for feature in importances:
        print feature