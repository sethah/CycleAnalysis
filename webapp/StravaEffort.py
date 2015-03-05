import numpy as np
import pandas as pd
from SignalProc import weighted_average, smooth, diff, vel_to_time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from PlotTools import PlotTool
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
        self.is_route = is_route
        self.belongs_to = belongs_to
        if belongs_to == 'other':
            self.init_from_other(activity_id, athlete_id, get_streams)
        else:
            d = self.fetch_activity(activity_id, athlete_id, belongs_to)
            self.name = d['name']
            self.dt = d['start_dt']
            self.total_distance = d['distance']
            self.moving_time = int(d.get('moving_time', 0))
            self.moving_time_string = time.strftime('%H:%M:%S', time.gmtime(self.moving_time))
            self.city = d['city']
            self.total_distance = d['distance']
            self.total_climb = d['total_elevation_gain']

            self.fitness10 = d['fitness10']
            self.fitness30 = d['fitness30']
            self.frequency10 = d['frequency10']
            self.frequency30 = d['frequency30']

            if get_streams:
                self.init_streams()
                self.center = self.get_center()

    def fetch_activity(self, activity_id, athlete_id, belongs_to):
        DB = StravaDB()
        self.id = activity_id
        self.athlete = athlete_id
        
        if self.is_route:
            table = 'routes'
            cols = ['id', 'athlete_id', 'start_dt', 'name',
                'city', 'fitness10', 'fitness30', 'frequency10', 'frequency30',
                'total_elevation_gain', 'distance']
        else:
            table = 'activities'
            cols = ['id', 'athlete_id', 'start_dt', 'name', 'moving_time',
                'city', 'fitness10', 'fitness30', 'frequency10', 'frequency30',
                'total_elevation_gain', 'distance']

        if belongs_to == 'other':
            q = """ SELECT %s FROM %s WHERE id = %s
            """ % (', '.join(cols), table, self.id)
        else:
            q = """ SELECT %s FROM %s WHERE id = %s AND athlete_id = %s
            """ % (', '.join(cols), table, self.id, self.athlete)

        print q

        DB.cur.execute(q)
        results = DB.cur.fetchone()
        d = dict(zip(cols, results))

        return d

    def init_streams(self):
        DB = StravaDB()
        cols = ['activity_id', 'athlete_id', 'time', 'distance',
                'velocity', 'grade', 'altitude', 'latitude', 'longitude']
        q = """SELECT %s FROM streams WHERE activity_id = %s""" % (', '.join(cols), self.id)
        self.df = pd.read_sql(q, DB.conn)
        if self.is_route:
            self.df = self.df.sort('distance')

    def init_from_other(self, activity_id, athlete_id, get_streams, dt=None):
        d = self.fetch_activity(activity_id, athlete_id, 'other')
        self.name = d['name']
        if dt == None:
            self.dt = datetime.now().date()
        else:
            self.dt = dt
        self.total_distance = d['distance']
        self.moving_time = 0
        self.moving_time_string = time.strftime('%H:%M:%S', time.gmtime(self.moving_time))
        self.city = d['city']
        self.total_distance = d['distance']
        self.total_climb = d['total_elevation_gain']

        self.get_my_fitness()

        if get_streams:
            self.init_streams()
            self.df.time = np.arange(-1, -self.df.shape[0] - 1, -1)
            self.df.velocity = np.array([-1]*self.df.shape[0])
            self.center = self.get_center()

    def strava_date(self, date_string):
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')

    def init_stream(self, stream_dict, stream_type):
        if stream_type not in stream_dict:
            return None
        return StravaStream(stream_type, stream_dict[stream_type])

    def get_moving(self):
        not_moving = np.where(~self.moving.raw_data)[0]
        for ind in not_moving:
            self.time.raw_data[ind:] -= (self.time.raw_data[ind] - \
                                         self.time.raw_data[ind -1])

        dd = np.diff(self.df.distance)
        dt = np.diff(self.df.time)
        not_moving = np.where(dd/dt < 1)[0]
        for ind in not_moving:
            ind += 1
            self.df.time.iloc[ind:] -= (self.df.time.iloc[ind] - \
                                         self.df.time.iloc[ind -1])

    def predict(self, model):
        df = self.make_df()

        df.pop('velocity')
        X = df.values
        pred = model.predict(X)        
        self.df['predicted_velocity'] = pred
        self.df['predicted_time'] = vel_to_time(self.df.predicted_velocity, self.df.distance)
        self.predicted_moving_time = time.strftime('%H:%M:%S', time.gmtime(pred[-1]))

    def streaming_predict(self):

        t = (self.df.time - self.df.time.iloc[0]).values
        pt = self.df.predicted_time.values

        predict = t - pt + pt[-1]
        return predict


    def to_dict(self):
        js = {}
        js['name'] = self.name
        js['date'] = datetime.strftime(self.dt, '%A %B %d, %Y')
        
        js['athlete'] = self.athlete
        pt = self.df.predicted_time.values.tolist()
        stream_predict = self.streaming_predict()
        
        # time, distance, altitude in the correct units
        d = (self.df.distance.values - self.df.distance.values[0]) / meters_per_mile
        t = (self.df.time.values - self.df.time.values[0])
        alt = (self.df.altitude.values * feet_per_meter).tolist()

        

        num_samples = 5000
        if self.is_route or self.belongs_to == 'other':
            # we convert the time to evenly spaced samples of length num_samples
            new_t = np.linspace(0, pt[-1], num_samples)
            js['type'] = 'route'

            # convert the predicted distance and altitude to the new time axis
            js['plot_predicted_distance'] = np.interp(new_t, pt, d).tolist()
            js['plot_predicted_altitude'] = np.interp(new_t, pt, alt).tolist()
            js['ride_rating'] = [0, 'NA']
            js['moving_time'] = 0
            js['moving_time_string'] = 'NA'
            js['start_time'] = 'NA'
        else:
            # for the ride simulation we need to go until the time when both have finished
            max_time = np.max([t[-1], pt[-1]])
            new_t = np.linspace(0, max_time, num_samples)

            # interpolate the predictions to have the same time axis
            pd = np.interp(t, pt, d)
            palt = np.interp(t, pt, alt)

            # convert the predicted distance and altitude to the new time axis
            js['plot_predicted_distance'] = np.interp(new_t, t, pd).tolist()
            js['plot_predicted_altitude'] = np.interp(new_t, t, palt).tolist()
            
            js['type'] = 'activity'
            js['ride_rating'] = self.ride_score()
            js['moving_time'] = t[-1]
            js['start_time'] = datetime.strftime(self.dt, '%H:%M:%S %p')
            js['moving_time_string'] = time.strftime('%H:%M:%S', time.gmtime(t[-1]))

        # convert the actual distance and altitude to the new time axis
        js['plot_time'] = new_t.tolist()
        js['plot_distance'] = np.interp(new_t, t, d).tolist()
        js['plot_altitude'] = np.interp(new_t, t, alt).tolist()
        js['streaming_predict'] = smooth(np.interp(new_t, t, stream_predict), 'scipy', window_len=200).tolist()
        
        # it doesn't matter that the latlng values do not correspond to the other vectors
        # because of the way the google maps markers are moved on a polyline
        js['latitude'] = self.df.latitude.values.tolist()
        js['longitude'] = self.df.longitude.values.tolist()
        js['center'] = self.get_center().tolist()
        
        js['total_distance'] = self.total_distance / meters_per_mile
        js['predicted_total_time'] = time.strftime('%H:%M:%S', time.gmtime(pt[-1]))
        
        
        js['id'] = self.id
        

        return js

    def ride_score(self):
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
        DB = StravaDB()
        dt = self.dt

        cols = ['id', 'start_dt', 'distance', 'total_elevation_gain']

        q = """ SELECT %s
                FROM activities
                WHERE start_dt >= '%s'
                AND start_dt < '%s'
                AND athlete_id = %s
            """ % (', '.join(cols), dt - timedelta(30), dt, self.athlete)

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

    def get_center(self):
        max_left = np.min(self.df.latitude)
        max_right = np.max(self.df.latitude)
        max_down = np.min(self.df.longitude)
        max_up = np.max(self.df.longitude)

        ne = [max_left, max_up]
        sw = [max_right, max_down]

        return np.array([sw, ne])

    def fitness_level(self, level):
        self.fit_level = level

    def make_df(self, window=6):
        n = self.df.shape[0]
        df = self.df.copy()
        df.pop('latitude')
        df.pop('longitude')
        # df.pop('velocity')
        df.pop('activity_id')
        df.pop('athlete_id')
        df['grade'] = smooth(df['grade'], 'scipy')
        df['altitude'] = smooth(df['altitude'], 'scipy', window_len=22)
        df['velocity'] = smooth(df['velocity'], 'scipy', window_len=100)
        # df['time_int'] = np.append(np.diff(df['time']), 0)
        # df['dist_int'] = np.append(np.diff(df['distance']), 0)

        # self.df['filtered_grade'] = smooth(self.df['grade'], 'scipy')
        alt_diff = np.diff(df['altitude'])
        climb = np.cumsum(np.where(alt_diff < 0, 0, alt_diff))
        climb = np.append([0], climb)
        grade_smooth = smooth(self.df.grade, 'scipy', window_len=np.min([300, self.df.shape[0]]))
        df['grade_smooth'] = grade_smooth
        grade_very_smooth = smooth(self.df.grade, 'scipy', window_len=np.min([500, self.df.shape[0]]))
        df['grade_very_smooth'] = grade_very_smooth
        df['climb'] = climb
        df['time'] = df['time'] - df['time'].iloc[0]
        df['distance'] = df['distance'] - df['distance'].iloc[0]
        df['ride_difficulty'] = [df['distance'].iloc[-1]*climb[-1]]*n
        df['fitness10'] = [self.fitness10]*n
        df['fitness30'] = [self.fitness30]*n
        df['frequency10'] = [self.frequency10]*n
        df['frequency30'] = [self.frequency30]*n

        # if window != 0:
        #     for i in xrange(-window // 2, window // 2 + 1):
        #         if i == 0:
        #             continue
        #         df['%s_%s' % ('grade', -i)] = df['grade'].shift(i)

        #     df.rename(columns={'grade': 'grade_0'}, inplace=True)
        df.pop('time')
        # df.pop('ride_difficulty')
        # df.pop('distance')
        df.pop('altitude')
        df.fillna(0, inplace=True)
        return df

    def __repr__(self):
        return '<%s, %s, %s>' % \
            (self.name, self.dt, self.city)


if __name__ == '__main__':
    pass