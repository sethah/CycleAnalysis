import numpy as np
import pandas as pd
import pymongo
from SignalProc import weighted_average, smooth, diff
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from datetime import datetime, timedelta
from StravaEffort import StravaActivity
from StravaDB import StravaDB
from StravaAPI import StravaAPI
import matplotlib.pyplot as plt
from PlotTools import PlotTool
import seaborn as sns
import pickle
from itertools import izip

# global vars
meters_per_mile = 1609.34
feet_per_meter = 3.280


# CLIENT = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
DB = StravaDB()

class StravaUser(object):

    def __init__(self, athlete_id, get_streams=False):
        self.userid = athlete_id
        self.init()
        self.load_activities(get_streams)
        # self.recent_fitness_level()

    def init(self):
        cols = ['id', 'firstname', 'lastname', 'sex', 'city', 'state', 'country', 'access_token']
        q = """SELECT * FROM athletes WHERE id = %s""" % self.userid
        DB.cur.execute(q)
        athlete = DB.cur.fetchone()
        athlete = {col: val for col, val in zip(cols, athlete)}
        self.name = athlete['firstname'] + ' ' + athlete['lastname']
        self.access_token = athlete['access_token']
        if 'model' in athlete:
            self.model = pickle.loads(athlete['model'])


    def load_activities(self, get_streams, min_length=990):

        # find all the activities in the db for this user
        if get_streams:
            query = {'athlete.id': self.userid}
            activities = list(DB.activities.find(query))
        else:
            # if we don't need the streams, don't query on them
            cols = ['id', 'athlete_id', 'start_dt', 'name', 'moving_time',
                    'city', 'fitness_level', 'total_elevation_gain', 'distance']
            q = """ SELECT %s FROM activities WHERE athlete_id = %s
                """ % (', '.join(cols), 4478600)
            results = DB.execute(q)

        self.activities = []
        for activity in results:
            d = dict(zip(cols, activity))
            a = StravaActivity(d, get_streams)
            self.activities.append(a)

    def has_full_predictions(self):
        if self.activities is None:
            return False

        for a in self.activities:
            if not a.has_prediction:
                return False

        return True

    def get_activities(self):
        api = StravaAPI(self.access_token)
        print 'Storing activities...............'
        api.store_activities()
        print 'Got \'em!'

    def recent_fitness_level(self):
        for a in self.activities:
            level = np.sum([x.total_distance*x.total_climb \
                    for x in self.activities \
                    if ((x.dt >= a.dt - timedelta(30)) and (x.dt < a.dt))])
            level = np.min([1e9, level])
            level /= 1e9
            a.fitness_level(level)

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
                df = a.make_df()
            else:
                df = df.append(a.make_df(), ignore_index=True)

        return df


if __name__ == '__main__':
    u = StravaUser('Seth')
    df = u.make_df((0,30))
    df = df[df['velocity'] > 3]
    y = df.pop('velocity')
    # df.pop('altitude')
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    # rf.score(X_test, y_test)
    # cvscore = cross_val_score(rf, X_train, y_train, cv=10, scoring='r2')
    # print np.mean(cvscore)
    # importances = sorted(zip(rf.feature_importances_, df.columns),
    #                      key=lambda x: x[0], reverse=True)
    # for feature in importances:
    #     print feature