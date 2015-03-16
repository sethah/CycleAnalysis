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
import seaborn as sns
import pickle
from itertools import izip

# global vars
meters_per_mile = 1609.34
feet_per_meter = 3.280


class StravaUser(object):

    def __init__(self, athlete_id, get_streams=False, get_routes=True):
        """
        INPUT: StravaUser, INT, BOOL, BOOL
        OUTPUT: None
        Initialize a Strava user.

        athlete_id is an integer id for a Strava app athlete.
        get_streams is a Boolean which indicates whether to get the raw data
        streams for the user.
        get_routes is a Boolean which indicates whether to load the user's
        routes as activities.
        """
        self.userid = athlete_id
        self.init()
        self.load_activities(get_streams, get_routes)

    def init(self):
        """
        INPUT: StravaUser
        OUTPUT: None

        Get the user's attributes from the database.
        """
        DB = StravaDB()
        cols = ['id', 'firstname', 'lastname', 'sex', 'city',
                'state', 'country', 'access_token']
        q = """SELECT * FROM athletes WHERE id = %s""" % self.userid
        DB.cur.execute(q)
        athlete = DB.cur.fetchone()
        athlete = {col: val for col, val in zip(cols, athlete)}
        self.firstname = athlete['firstname']
        self.lastname = athlete['lastname']
        self.name = self.firstname + ' ' + self.lastname
        self.access_token = athlete['access_token']
        if 'model' in athlete:
            self.model = pickle.loads(athlete['model'])

    def load_activities(self, get_streams, get_routes):
        """
        INPUT: StravaUser, BOOL, BOOL
        OUTPUT: None

        Populate the activities attribute as a list of StravaActivity objects.
        """

        # find all the activities in the db for this user
        DB = StravaDB()
        q = """ SELECT id
                FROM activities
                WHERE athlete_id = %s
            """ % self.userid
        results = DB.execute(q)

        self.activities = []
        for activity in results:
            a = StravaActivity(activity[0], self.userid, get_streams)
            self.activities.append(a)

        if get_routes:
            # get the routes for this user also
            q = """ SELECT id
                    FROM routes
                    WHERE athlete_id = %s
                """ % self.userid
            results = DB.execute(q)
            for route in results:
                r = StravaActivity(route[0], self.userid, get_streams,
                                   is_route=True)
                self.activities.append(r)

    def get_activities(self):
        """
        INPUT: StravaUser
        OUTPUT: None

        Use the user's auth token to get their activities from the
        Strava API.
        """
        api = StravaAPI(self.access_token)
        print 'Storing activities...............'
        api.store_activities()
        print 'Got \'em!'

    def make_df(self, indices=None):
        """
        INPUT: StravaUser, 1D NUMPY ARRAY
        OUTPUT: None

        Make a dataframe of time samples to be used for predictions.

        indices is a 1D numpy array which indicates which activities to
        use for predictions.
        """
        if indices is None:
            indices = np.arange(len(self.activities))

        df = None
        for ix in indices:
            a = self.activities[ix]
            if df is None:
                df = a.make_df()
            else:
                df = df.append(a.make_df(), ignore_index=True)

        return df
