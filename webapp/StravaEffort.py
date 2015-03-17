import numpy as np
import pandas as pd
from SignalProc import weighted_average, smooth, diff, vel_to_time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from StravaDB import StravaDB
from datetime import datetime, timedelta, date
import pymongo
import seaborn as sns
import time
import gpxpy

# global vars
meters_per_mile = 1609.34
feet_per_meter = 3.280

DB = StravaDB()


class StravaActivity(object):

    def __init__(self, activity_id, athlete_id, get_streams=False, is_route=False, belongs_to='athlete'):
        """
        INPUT: StravaActivity, INT, INT, BOOL, BOOL, STRING
        OUTPUT: None

        Initialize a StravaActivity

        activity_id is the integer id assigned to this activity by Strava. If the
        activity is a route, then this id was assigned automatically by the database.

        athlete_id is the integer id of the parent StravaUser

        get_streams is a Boolean which indicates a Boolean which indicates whether to get the raw data
        streams for the user.

        is_route is a Boolean which indicates if this is a ride that the user has not done
        yet and thus does not have time/velocity streams.

        belongs_to is a Boolean which indicates whether this activity belongs to the parent
        StravaUser. If belongs_to == 'other', then this athlete does not have time/velocity
        streams and the activity is treated like a route.
        """
        self.is_route = is_route
        self.belongs_to = belongs_to
        self.id = activity_id
        self.athlete_id = athlete_id
        self.athlete = self.get_athlete()
        if self.belongs_to == 'other':
            self.init_from_other(get_streams)
        else:
            d = self.fetch_activity()
            self.name = d['name']  # description of the ride
            self.dt = d['start_dt']  # timestamp of the ride's start
            self.total_distance = d['distance']
            self.moving_time = int(d.get('moving_time', 0))
            self.moving_time_string = time.strftime('%H:%M:%S', time.gmtime(self.moving_time))
            self.city = d['city']
            self.total_distance = d['distance']
            self.total_climb = d['total_elevation_gain']
            self.athlete_count = d['athlete_count']

            self.get_my_fitness()

            if get_streams:
                self.init_streams()
                self.center = self.get_bounds()

    def get_athlete(self):
        """
        INPUT: StravaActivity
        OUTPUT: DICTIONARY

        Get the athlete from the database.
        """
        DB = StravaDB()
        cols = ['id', 'firstname', 'lastname', 'sex', 'city',
                'state', 'country']
        q = """ SELECT %s
                FROM athletes
                WHERE id = %s
            """ % (', '.join(cols), self.athlete_id)

        DB.cur.execute(q)
        athlete = DB.cur.fetchone()

        return dict(zip(cols, athlete))

    def fetch_activity(self):
        """
        INPUT: StravaActivity, INT, INT, BOOL, STRING
        OUTPUT: DICTIONARY

        Get an activity from the database.
        """
        DB = StravaDB()

        if self.is_route:
            table = 'routes'
            cols = ['id', 'athlete_id', 'start_dt', 'name',
                    'city', 'fitness10', 'fitness30', 'frequency10',
                    'frequency30', 'total_elevation_gain', 'distance',
                    'athlete_count']
        else:
            table = 'activities'
            cols = ['id', 'athlete_id', 'start_dt', 'name', 'moving_time',
                    'city', 'fitness10', 'fitness30', 'frequency10',
                    'frequency30', 'total_elevation_gain', 'distance',
                    'athlete_count']

        if self.belongs_to == 'other':
            q = """ SELECT %s FROM %s WHERE id = %s
            """ % (', '.join(cols), table, self.id)
        else:
            q = """ SELECT %s FROM %s WHERE id = %s AND athlete_id = %s
            """ % (', '.join(cols), table, self.id, self.athlete_id)

        DB.cur.execute(q)
        results = DB.cur.fetchone()

        return dict(zip(cols, results))

    def init_streams(self):
        """
        INPUT: StravaActivity
        OUTPUT: None

        Load the activity's streams into a Pandas dataframe.
        """
        DB = StravaDB()
        cols = ['activity_id', 'athlete_id', 'time', 'distance',
                'velocity', 'grade', 'altitude', 'latitude', 'longitude']
        q = """SELECT %s FROM streams WHERE activity_id = %s""" % (', '.join(cols), self.id)
        self.df = pd.read_sql(q, DB.conn)
        try:
            self.moving_time = self.df.time.iloc[-1] - self.df.time.iloc[0]
        except:
            raise
        if self.is_route:
            self.df = self.df.sort('distance')

    def init_from_other(self, get_streams, dt=None):
        """
        INPUT: StravaActivity, INT, INT, BOOL, DATETIME DATE
        OUTPUT: None

        Initialize an activity for the parent athlete that belongs to another
        athlete.

        This method is used to create what is essentially a route from an activity
        that another user has done, but this user has not. There is no moving data.
        """
        d = self.fetch_activity()
        self.name = d['name']
        self.dt = datetime.now()
        self.total_distance = d['distance']
        self.moving_time = 0
        self.moving_time_string = time.strftime('%H:%M:%S', time.gmtime(self.moving_time))
        self.city = d['city']
        self.total_distance = d['distance']
        self.total_climb = d['total_elevation_gain']
        self.athlete_count = 1

        self.get_my_fitness()

        if get_streams:
            self.init_streams()
            self.df.time = np.arange(-1, -self.df.shape[0] - 1, -1)
            self.df.velocity = np.array([-1]*self.df.shape[0])
            self.center = self.get_bounds()

    def strava_date(self, date_string):
        """
        INPUT: StravaActivity, STRING
        OUTPUT: DATETIME

        Convert a Strava date string to datetime object.
        """
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')

    def predict(self, model):
        """
        INPUT: StravaActivity, SKLEARN MODEL
        OUTPUT: None

        Construct feature dataframe and predict time and velocity.

        model is a scikit-learn model object.
        """
        df = self.make_df()

        df.pop('velocity')
        X = df.values
        pred = model.predict(X)
        self.df['predicted_velocity'] = pred
        self.df['predicted_time'] = vel_to_time(self.df.predicted_velocity, self.df.distance)
        self.predicted_moving_time = time.strftime('%H:%M:%S', time.gmtime(self.df['predicted_time'].iloc[-1]))

    def streaming_predict(self):
        """
        INPUT: StravaActivity
        OUTPUT: 1D NUMPY ARRAY

        Return a streaming prediction of the user for every time
        sample. This is the prediction if it were updated with the
        user's real tim data. It should converge to the true value.
        """

        t = (self.df.time - self.df.time.iloc[0]).values
        pt = self.df.predicted_time.values

        predict = t - pt + pt[-1]
        return predict

    def to_dict(self, time_spacing=None):
        """
        INPUT: StravaActivity, FLOAT
        OUTPUT: None

        Convert the activity to a dictionary with uniformly
        spaced time vectors for actual and predictions.

        time_spacing indicates the space between time samples.

        This method interpolates the values so that the predicted and
        actual time vectors have the same spacing. They are not the same
        length, however.
        """
        js = {}
        js['name'] = self.name
        js['date'] = datetime.strftime(self.dt, '%A %B %d, %Y')

        js['athlete'] = self.athlete_id
        pt = self.df.predicted_time.values.tolist()
        stream_predict = self.streaming_predict()

        # time, distance, altitude in the correct units
        d = (self.df.distance.values - self.df.distance.values[0]) / meters_per_mile
        t = (self.df.time.values - self.df.time.values[0])
        alt = (self.df.altitude.values * feet_per_meter).tolist()
        v = (self.df.velocity.values) / meters_per_mile * 3600

        num_samples = 5000
        if self.is_route or self.belongs_to == 'other':
            # we convert the time to evenly spaced samples

            if time_spacing is None:
                new_time = np.arange(0, pt[-1], pt[-1]/float(num_samples))
            else:
                new_time = np.arange(0, pt[-1], time_spacing)

            js['type'] = 'route'

            # convert the predicted distance and altitude to the new time axis
            js['plot_predicted_distance'] = np.interp(new_time, pt, d).tolist()
            js['plot_predicted_altitude'] = np.interp(new_time, pt, alt).tolist()
            js['ride_rating'] = [0, 'NA']
            js['moving_time'] = 0
            js['moving_time_string'] = 'NA'
            js['start_time'] = 'NA'
        else:
            # for the ride simulation we need to go until the time when both have finished
            if time_spacing is None:
                min_time = np.min([t[-1], pt[-1]])
                spacing = min_time / num_samples
                new_time_predicted = np.arange(0, pt[-1], spacing)
                new_time = np.arange(0, t[-1], spacing)
            else:
                new_time = np.arange(0, t[-1], time_spacing)
                new_time_predicted = np.arange(0, pt[-1], time_spacing)

            # convert the predicted distance and altitude to the new time axis
            js['plot_predicted_distance'] = np.interp(new_time_predicted, pt, d).tolist()
            js['plot_predicted_altitude'] = np.interp(new_time_predicted, pt, alt).tolist()
            js['plot_predicted_velocity'] = np.interp(new_time_predicted, pt, self.df.predicted_velocity.values / meters_per_mile * 3600).tolist()

            js['type'] = 'activity'
            js['ride_rating'] = self.ride_score()
            js['moving_time'] = t[-1]
            js['start_time'] = datetime.strftime(self.dt, '%H:%M:%S %p')
            js['moving_time_string'] = time.strftime('%H:%M:%S', time.gmtime(t[-1]))

        # convert the actual distance and altitude to the new time axis
        js['plot_time'] = new_time.tolist()
        js['plot_distance'] = np.interp(new_time, t, d).tolist()
        js['plot_altitude'] = np.interp(new_time, t, alt).tolist()
        js['plot_velocity'] = np.interp(new_time, t, v).tolist()
        js['streaming_predict'] = smooth(np.interp(new_time, t, stream_predict), 'scipy', window_len=200).tolist()

        # it doesn't matter that the latlng values do not correspond to the other vectors
        # because of the way the google maps markers are moved on a polyline
        js['latitude'] = self.df.latitude.values.tolist()
        js['longitude'] = self.df.longitude.values.tolist()
        js['center'] = self.get_bounds().tolist()

        js['total_distance'] = self.total_distance / meters_per_mile
        js['predicted_total_time'] = pt[-1]
        js['predicted_total_time_string'] = time.strftime('%H:%M:%S', time.gmtime(pt[-1]))
        js['id'] = self.id

        return js

    def to_dict2(self, time_spacing=None):
        """
        INPUT: StravaActivity, FLOAT
        OUTPUT: None

        Convert the activity to a dictionary with uniformly
        spaced time vectors for actual and predictions.

        time_spacing indicates the space between time samples.

        This method interpolates the values so that the predicted and
        actual time vectors have the same spacing. They are not the same
        length, however.
        """

        predicted = {}
        actual = {}
        predicted['name'] = self.name
        predicted['date'] = datetime.strftime(self.dt, '%A %B %d, %Y')
        predicted['athlete'] = self.athlete
        predicted['latitude'] = self.df.latitude.values.tolist()
        predicted['longitude'] = self.df.longitude.values.tolist()
        predicted['center'] = self.get_bounds().tolist()
        predicted['id'] = self.id

        pt = self.df.predicted_time.values.tolist()
        stream_predict = self.streaming_predict()

        # time, distance, altitude in the correct units
        d = (self.df.distance.values - self.df.distance.values[0]) / meters_per_mile
        alt = (self.df.altitude.values * feet_per_meter).tolist()

        num_samples = 5000
        if self.is_route or self.belongs_to == 'other':
            # we convert the time to evenly spaced samples

            if time_spacing is None:
                new_time = np.arange(0, pt[-1], pt[-1]/float(num_samples))
            else:
                new_time = np.arange(0, pt[-1], time_spacing)

            predicted['type'] = 'route'

            # convert the predicted distance and altitude to the new time axis
            predicted['plot_time'] = new_time.tolist()
            predicted['plot_distance'] = np.interp(new_time, pt, d).tolist()
            predicted['plot_altitude'] = np.interp(new_time, pt, alt).tolist()
            predicted['ride_rating'] = [0, 'NA']
            predicted['moving_time'] = pt[-1]
            predicted['moving_time_string'] = time.strftime('%H:%M:%S', time.gmtime(pt[-1]))
            predicted['start_time'] = 'NA'
            predicted['total_distance'] = self.total_distance / meters_per_mile
            predicted['total_elevation_gain'] = self.total_climb
            # predicted['athlete_count'] = self.athlete_count
        else:

            actual['name'] = self.name
            actual['date'] = datetime.strftime(self.dt, '%A %B %d, %Y')
            actual['athlete'] = self.athlete

            t = (self.df.time.values - self.df.time.values[0])
            v = (self.df.velocity.values) / meters_per_mile * 3600
            # for the ride simulation we need to go until the time when both have finished
            if time_spacing is None:
                min_time = np.min([t[-1], pt[-1]])
                spacing = min_time / num_samples
                new_time_predicted = np.arange(0, pt[-1], spacing)
                new_time = np.arange(0, t[-1], spacing)
            else:
                new_time = np.arange(0, t[-1], time_spacing)
                new_time_predicted = np.arange(0, pt[-1], time_spacing)

            actual['plot_time'] = new_time.tolist()
            actual['plot_distance'] = np.interp(new_time, t, d).tolist()
            actual['plot_altitude'] = np.interp(new_time, t, alt).tolist()
            actual['plot_velocity'] = np.interp(new_time, t, v).tolist()

            # convert the predicted distance and altitude to the new time axis
            predicted['plot_distance'] = np.interp(new_time_predicted, pt, d).tolist()
            predicted['plot_altitude'] = np.interp(new_time_predicted, pt, alt).tolist()
            predicted['plot_velocity'] = np.interp(new_time_predicted, pt, self.df.predicted_velocity.values / meters_per_mile * 3600).tolist()

            predicted['type'] = 'activity'
            predicted['ride_rating'] = self.ride_score()
            predicted['moving_time'] = pt[-1]
            predicted['start_time'] = datetime.strftime(self.dt, '%I:%M:%S %p')
            predicted['moving_time_string'] = time.strftime('%H:%M:%S', time.gmtime(pt[-1]))
            predicted['total_distance'] = self.total_distance / meters_per_mile
            predicted['plot_time'] = new_time_predicted.tolist()
            # predicted['athlete_count'] = self.athlete_count
            actual['streaming_predict'] = smooth(np.interp(new_time, t, stream_predict), 'scipy', window_len=200).tolist()
            actual['total_elevation_gain'] = self.total_climb
            # actual['athlete_count'] = self.athlete_count

            actual['type'] = 'activity'
            actual['ride_rating'] = self.ride_score()
            actual['moving_time'] = t[-1]
            actual['start_time'] = datetime.strftime(self.dt, '%I:%M:%S %p')
            actual['moving_time_string'] = time.strftime('%H:%M:%S', time.gmtime(t[-1]))
            actual['total_distance'] = self.total_distance / meters_per_mile
            actual['plot_time'] = new_time.tolist()

            actual['latitude'] = self.df.latitude.values.tolist()
            actual['longitude'] = self.df.longitude.values.tolist()
            actual['center'] = self.get_bounds().tolist()
            actual['id'] = self.id

        return actual, predicted

    def ride_score(self):
        """
        INPUT: StravaActivity
        OUTPUT: TUPLE

        Return a tuple of numeric and string rating of the ride.

        This method compares the actual time to the predicted time
        to evaluate and score a rider's performance.
        """
        ratings = ['Poor', 'Below Average', 'Average', 'Good', 'Great!', 'Excellent']
        scores = range(len(ratings))
        criteria = np.array([0.5, 0.3, 0, -0.2, -0.3, -0.4])
        if self.predicted_moving_time is not None:
            predicted_time = self.df.predicted_time.iloc[-1]
            print predicted_time, self.moving_time
            performance = (self.moving_time - predicted_time) / predicted_time

            tmp = criteria-performance
            ind = np.argmin(np.where(tmp < 0, 999, tmp))

        return (scores[ind], ratings[ind])

    def get_my_fitness(self):
        """
        INPUT: StravaActivity
        OUTPUT: None

        Get the fitness/frequency features for the activity. If the activity
        belongs to the user then these are already populated in the database.
        This method is used when the activity is a route.
        """
        DB = StravaDB()
        dt = self.dt.date()

        cols = ['id', 'start_dt', 'distance', 'total_elevation_gain']

        q = """ SELECT %s
                FROM activities
                WHERE start_dt >= '%s'
                AND start_dt < '%s'
                AND athlete_id = %s
            """ % (', '.join(cols), dt - timedelta(30), dt, self.athlete_id)

        results = DB.execute(q)
        difficulties10 = []
        difficulties30 = []
        for a in results:
            d = dict(zip(cols, a))
            dt2 = d['start_dt'].date()
            if dt2 < dt and dt2 >= dt - timedelta(30):
                difficulty = d['total_elevation_gain']*d['distance']
                difficulties30.append(difficulty)
            if dt2 < dt and dt2 >= dt - timedelta(10):
                difficulty = d['total_elevation_gain']*d['distance']
                difficulties10.append(difficulty)
        self.fitness10 = np.sum(difficulties10)
        self.fitness30 = np.sum(difficulties30)
        self.frequency10 = len(difficulties10)
        self.frequency30 = len(difficulties30)

    def get_bounds(self):
        """
        INPUT: StravaActivity
        OUTPUT: 2D NUMPY ARRAY

        Return the SW and NE lat/lng bounds for the ride.
        """
        max_left = np.min(self.df.latitude)
        max_right = np.max(self.df.latitude)
        max_down = np.min(self.df.longitude)
        max_up = np.max(self.df.longitude)

        ne = [max_left, max_up]
        sw = [max_right, max_down]

        return np.array([sw, ne])

    def distance_rolling_mean(self, mean_col, length=1):

        new_distance = np.linspace(0, self.df.distance.iloc[-1], self.df.shape[0])
        new_col = np.interp(new_distance, self.df.distance, self.df[mean_col])
        dx = new_distance[1] - new_distance[0]
        n = int(length * 1609.34 / dx)
        rolling_mean = pd.stats.moments.rolling_mean(new_col, n)
        rolling_mean = np.nan_to_num(rolling_mean)
        rolling_mean_converted = np.interp(self.df.distance, new_distance, rolling_mean)
        return rolling_mean_converted

    def make_df(self):
        """
        INPUT: StravaActivity
        OUTPUT: None

        Construct the feature/target dataframe for the activity.
        """
        n = self.df.shape[0]
        df = self.df.copy()
        if 'predicted_time' in df.columns:
            df.pop('predicted_time')
            df.pop('predicted_velocity')
        df.pop('latitude')
        df.pop('longitude')

        df.pop('activity_id')
        df.pop('athlete_id')
        df['grade'] = smooth(df['grade'], 'scipy')
        df['altitude'] = smooth(df['altitude'], 'scipy', window_len=22)
        df['velocity'] = smooth(df['velocity'], 'scipy', window_len=100)

        alt_diff = np.diff(df['altitude'])
        climb = np.cumsum(np.where(alt_diff < 0, 0, alt_diff))
        climb = np.append([0], climb)
        grade_smooth = smooth(self.df.grade, 'scipy', window_len=np.min([20, self.df.shape[0]]))
        df['grade_smooth'] = grade_smooth
        grade_very_smooth = smooth(self.df.grade, 'scipy', window_len=np.min([50, self.df.shape[0]]))
        df['grade_very_smooth'] = grade_very_smooth
        df['climb'] = climb
        df['time'] = df['time'] - df['time'].iloc[0]
        df['distance'] = df['distance'] - df['distance'].iloc[0]
        df['ride_difficulty'] = [df['distance'].iloc[-1]*climb[-1]]*n
        df['variability'] = [df['grade'].std()]*n
        df['fitness10'] = [self.fitness10]*n
        df['fitness30'] = [self.fitness30]*n
        df['frequency10'] = [self.frequency10]*n
        df['frequency30'] = [self.frequency30]*n
        df['one_mile'] = self.distance_rolling_mean('grade')
        df['recent'] = self.distance_rolling_mean('grade', 0.1)

        df.pop('time')
        df.pop('altitude')
        df.fillna(0, inplace=True)
        return df

    def __repr__(self):
        return '<%s, %s, %s, %s>' % \
            (self.name, self.dt, self.city, self.total_distance)
