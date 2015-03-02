import numpy as np
import pandas as pd
from SignalProc import weighted_average, smooth, diff
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from PlotTools import PlotTool
from StravaDB import StravaDB
from datetime import datetime
import pymongo
import seaborn as sns
import time

# global vars
meters_per_mile = 1609.34
feet_per_meter = 3.280

DB = StravaDB()

class StravaActivity(object):

    def __init__(self, activity_id, get_streams=False):
        self.id = activity_id
        cols = ['id', 'athlete_id', 'start_dt', 'name', 'moving_time',
                'city', 'fitness_level', 'total_elevation_gain', 'distance']
        q = """ SELECT %s FROM activities WHERE id = %s
            """ % (', '.join(cols), self.id)
        DB.cur.execute(q)
        results = DB.cur.fetchone()
        d = dict(zip(cols, results))
        self.name = d['name']
        self.athlete = int(d['athlete_id'])
        self.dt = self.strava_date(d['start_dt'])
        self.total_distance = d['distance']
        self.moving_time = int(d['moving_time'])

        if get_streams:
            self.init_streams()

        self.city = d['city']
        self.total_distance = d['distance']
        self.fitness_level = d['fitness_level']
        self.total_climb = d['total_elevation_gain']

    def init_streams(self):
        cols = ['activity_id', 'athlete_id', 'time', 'distance',
                'velocity', 'grade', 'altitude', 'latitude', 'longitude']
        q = """SELECT %s FROM streams WHERE activity_id = %s""" % (', '.join(cols), self.id)
        self.df = pd.read_sql(q, DB.db)

    def strava_date(self, date_string):
        if type(date_string) != unicode:
            return date_string
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')

    def init_stream(self, stream_dict, stream_type):
        if stream_type not in stream_dict:
            return None
        return StravaStream(stream_type, stream_dict[stream_type])

    def is_ride(self, velocity):
        min_vel = 4.  # mph
        max_vel = 30.  # mph

        avg_vel = np.mean(velocity)
        return avg_vel >= min_vel and avg_vel <= max_vel

    def get_moving(self):
        not_moving = np.where(~self.moving.raw_data)[0]
        for ind in not_moving:
            self.time.raw_data[ind:] -= (self.time.raw_data[ind] - \
                                         self.time.raw_data[ind -1])

        dd = np.diff(self.distance.raw_data)
        dt = np.diff(self.time.raw_data)
        not_moving = np.where(dd/dt < 1)[0]
        for ind in not_moving:
            ind += 1
            self.time.raw_data[ind:] -= (self.time.raw_data[ind] - \
                                         self.time.raw_data[ind -1])

        self.time.filter()

    def predict(self, model):
        df = self.make_df()
        print df.info()
        print df.head()
        df.pop('time_int')
        X = df.values
        pred_int = model.predict(X)
        pred = np.cumsum(pred_int)
        self.predicted_time = StravaStream('predicted_time', {'data': pred})
        self.predicted_moving_time = pred[-1]

    def to_dict(self):
        js = {}
        js['name'] = self.name
        js['date'] = datetime.strftime(self.dt.date(), '%A %B %d, %Y')
        js['start_time'] = datetime.strftime(self.dt, '%H:%M:%S %p')
        js['athlete'] = self.athlete['id']
        d = (self.distance.raw_data - self.time.raw_data[0]) / meters_per_mile
        t = (self.time.raw_data - self.time.raw_data[0])
        step = t[-1] / (1000)
        pt = self.predicted_time.raw_data
        new_t = np.arange(0, max(pt[-1], t[-1]) + step, step)

        js['altitude'] = (np.interp(new_t, t, self.altitude.filtered) * feet_per_meter).tolist()
        js['altitude_interp'] = np.interp(new_t, pt, self.altitude.filtered * feet_per_meter).tolist()
        js['distance'] = d.tolist()
        js['distance_interp'] = np.interp(new_t, t, d).tolist()
        # js['velocity'] = self.velocity.filtered.tolist()
        js['latlng'] = self.latlng.raw_data[np.arange(0, self.latlng.raw_data.shape[0], 4)].tolist()
        js['performance_rating'] = self.rating
        js['center'] = self.get_center().tolist()
        # js['time'] = t.tolist()
        js['time_interp'] = new_t.tolist()
        js['predicted_time'] = pt.tolist()
        js['predicted_distance'] = np.interp(new_t, pt, d).tolist()
        js['total_distance'] = self.total_distance
        js['predicted_total_time'] = time.strftime('%H:%M:%S', time.gmtime(pt[-1]))
        js['moving_time'] = time.strftime('%H:%M:%S', time.gmtime(self.moving_time))
        # js['grade'] = self.grade.filtered.tolist()
        js['id'] = self.id

        return js

    def ride_score(self):
        ratings = ['Poor', 'Below Average', 'Average', 'Good', 'Great!', 'Excellent']
        scores = range(len(ratings))
        criteria = np.array([0.3, 0.15, 0, -0.1, -0.2, -0.3])
        if self.predicted_moving_time is not None:
            predicted_time = self.predicted_moving_time
            performance = (self.moving_time - predicted_time) / predicted_time

            tmp = criteria-performance
            ind = np.argmin(np.where(tmp < 0, 999, tmp))

        return (scores[ind], ratings[ind])

    def get_center(self):
        latlng = self.latlng.raw_data

        max_left, max_up = np.max(latlng, axis=0)
        max_right, max_down = np.min(latlng, axis=0)

        ne = [max_left, max_up]
        sw = [max_right, max_down]

        return np.array([sw, ne])

    def hill_analysis(self):
        diff_alt = diff(self.altitude.raw_data[:, np.newaxis],
                        self.distance.raw_data[:, np.newaxis])
        diff_alt = smooth(diff_alt, 'scipy')

        hills = []
        hill_start = False
        hill_start_idx = 0
        hill_start_time = 0
        stream_length = self.distance.raw_data.shape[0]
        climb_total = np.zeros(stream_length)
        for k in xrange(stream_length):
            time_elapsed = self.time.raw_data[k] - hill_start_time
            if k != 0:
                climb = np.max([self.altitude.raw_data[k] - self.altitude.raw_data[k - 1], 0])
                climb_total[k] = climb_total[k - 1] + climb

            if diff_alt[k] > 0 and not hill_start:
                hill_start = True
                hill_start_idx = k
                hill_start_time = self.time.raw_data[hill_start_idx]

            elif hill_start and (diff_alt[k] < 0 or k == stream_length - 1):
                hill_start = False
                if len(hills) != 0:
                    prev_hill = hills[-1]
                else:
                    prev_hill = None
                if k - hill_start_idx >= 2:
                    stream_dict = self.make_stream_dict()
                    for key in stream_dict:
                        stream_dict[key]['data'] = stream_dict[key]['data'][hill_start_idx:k]

                    effort_dict = {'id': len(hills),
                                   'name': self.name,
                                   'athlete': self.athlete,
                                   'start_date': self.dt,
                                   'streams': stream_dict}
                    hill = StravaHill(effort_dict, self, previous_hill=prev_hill)
                    hills.append(hill)

        return self.clean_hills(hills)

    def make_stream_dict(self):
        return {stream_type: {'data': getattr(self, stream_type.replace('_smooth', '')).raw_data} \
                for stream_type in self.stream_types()}

    def clean_hills(self, hills):
        min_score = 8000. / meters_per_mile
        min_separation = 0.5  # miles
        clean_hills = []
        for k, hill in enumerate(hills):
            if len(clean_hills) != 0:
                if hill.distance.raw_data[0] - clean_hills[-1].distance.raw_data[-1] < min_separation:
                    hill = self.merge_hills(clean_hills[-1], hill)
                    clean_hills.pop()
            if hill.score > min_score:
                if hill.previous_hill is not None:

                    if hill.previous_hill.score < min_score:
                        hill.previous_hill = None
                clean_hills.append(hill)

        return clean_hills

    def merge_hills(self, hill1, hill2):
        stream_dict = {}
        for stream_type in self.stream_types():
            hill1_data = getattr(hill1, stream_type.replace('_smooth','')).raw_data
            hill2_data = getattr(hill2, stream_type.replace('_smooth','')).raw_data
            stream_dict[stream_type] = {}
            stream_dict[stream_type]['data'] = np.append(hill1_data, hill2_data)
        effort_dict = {'id': hill1.id,
                       'name': hill1.name,
                       'athlete': hill1.athlete,
                       'start_date': hill1.date,
                       'streams': stream_dict}
        hill = StravaHill(effort_dict, self, previous_hill=hill1.previous_hill)

        return hill

    def plot_hills(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        p = PlotTool()
        ax.plot(self.distance.raw_data, self.altitude.raw_data,
                c='r', label='altitude')
        xmin, xmax = np.min(self.distance.raw_data), np.max(self.distance.raw_data)
        ymin, ymax = np.min(self.altitude.raw_data), np.max(self.altitude.raw_data)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(self.dt.date())
        axis_label_font = 20
        ax.set_xlabel('Distance (miles)', fontsize=axis_label_font)
        ax.set_ylabel('Altitude (ft)', fontsize=axis_label_font)
        ax.set_title('%s, %s, %s' % \
                        (self.name, self.athlete['id'], self.dt.date()),
                        fontsize=24)

        ax.tick_params(labelsize=16)

        cmap = plt.get_cmap("autumn")

        for hill in self.hills:
            label = 'Score: %0.0f' % (hill.score)
            X = hill.distance.raw_data
            Y = hill.altitude.raw_data
            p.plot_fill(X, Y, hill.velocity.filtered, cmap, ax)

        plt.show()

    def plot_each_hill(self):
        p = PlotTool()
        r, c = p.subplot_dims(len(self.hills))
        fig, axs = plt.subplots(r, c, figsize=(15, 12))

        if r == 1 and c == 1:
            axs = np.array([axs])

        cmap = plt.get_cmap("autumn")
        for k, ax in enumerate(axs.reshape(-1)):
            hill = self.hills[k]
            label = 'Score: %0.0f' % (hill.score)
            X = hill.distance.raw_data
            Y = hill.altitude.raw_data
            p.plot_fill(X, Y, hill.velocity.filtered, cmap, ax)
            ax.set_xlabel('Distance (miles)')
            ax.set_ylabel('Altitude (feet)')
        fig.suptitle('Hills for %s on %s' % (self.name, self.dt.date()),
                     fontsize=20)
        plt.show()

    def plot_filter(self, key):
        stream = getattr(self, key)
        x = smooth(stream.raw_data, 'scipy')
        plt.plot(self.distance.raw_data, x, label='hanning')
        plt.plot(self.distance.raw_data, stream.filtered)

        plt.legend()
        plt.show()

    def hills_df(self):
        df = None
        for hill in self.hills:
            if df is None:
                df = hill.make_df()
            else:
                df = df.append(hill.make_df(), ignore_index=True)

        return df

    def fitness_level(self, level):
        self.fit_level = level

    def make_df(self, window=6):
        n = self.grade.filtered.shape[0]
        back, ahead = self.past_grade()
        alt_diff = np.diff(self.altitude.filtered-self.altitude.filtered[0])
        climb = np.cumsum(np.where(alt_diff < 0, 0, alt_diff))
        climb = np.append([0], climb)

        mytime = self.time.raw_data - self.time.raw_data[0]
        mydist = self.distance.raw_data - self.distance.raw_data[0]
        d = {'ride_difficulty': [self.distance.raw_data[-1]*climb[-1]]*n,
             'grade': self.grade.filtered,
             'climb': climb,
             # 'date': [time.mktime(self.dt.timetuple())]*n,
             'fitness_level': self.fit_level,
             'time_int': np.append(np.diff(mytime), 0),
             'dist_int': np.append(np.diff(mydist), 0),
             'distance': mydist,
             'time': mytime,
             'velocity': self.velocity.filtered,
             }
        df = pd.DataFrame(d)
        if window != 0:
            for i in xrange(-window // 2, window // 2 + 1):
                if i == 0:
                    continue
                df['%s_%s' % ('grade', -i)] = df['grade'].shift(i)

            df.pop('velocity')
            df.rename(columns={'grade': 'grade_0'}, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def __repr__(self):
        return '<%s, %s, %s>' % \
            (self.name, self.dt, self.city)


class StravaHill(object):
    """
    Possible features: variance of grade/altitude, date (getting better over time?)
    """

    def __init__(self, effort_dict, activity, previous_hill=None):
        self.init(effort_dict)
        self.activity = activity
        self.previous_hill = previous_hill
        self.score = self.hill_score()
        self.rating = np.mean(self.velocity.filtered)*self.score/50.
        self.dt = time.mktime(self.activity.date.timetuple())

    def hill_score(self):
        grade_vec = self.grade.raw_data
        dist_vec = self.distance.raw_data

        mean_grade = weighted_average(grade_vec, dist_vec)
        distance = self.distance.raw_data[-1] - self.distance.raw_data[0]

        return distance*mean_grade

    def hill_features(self):
        self.previous_climb = self.past_climb()
        self.time_elapsed = self.time.raw_data[0]
        self.distance_elapsed = self.distance.raw_data[0]
        if self.previous_hill is not None:
            self.time_since_last_climb = self.time.raw_data[0] - self.previous_hill.time.raw_data[-1]
            self.score_last_climb = self.previous_hill.score
        else:
            self.time_since_last_climb = 100000
            self.score_last_climb = 0

    def start_index(self):
        hill_start_time = self.time.raw_data[0]
        start = (np.abs(self.activity.time.raw_data-hill_start_time)).argmin()
        return start

    def past_climb(self):
        start = self.start_index()
        # print start
        diff = np.diff(self.activity.altitude.raw_data[0:start])
        return np.sum(np.where(diff < 0, 0, diff))

    def make_df(self):
        self.hill_features()
        n = self.grade.filtered.shape[0]
        d = {'activity_id': [self.activity.id]*n,
             'hill_id': [self.id]*n,
             'previous_climb': [self.previous_climb]*n,
             'time_elapsed': [self.time_elapsed]*n,
             'date': [self.dt]*n,
             'distance_elapsed': [self.distance_elapsed]*n,
             'time_since_last_climb': [self.time_since_last_climb]*n,
             'score_last_climb': [self.score_last_climb]*n,
             'grade': self.grade.filtered,
             'time': self.time.raw_data - self.time.raw_data[0],
             'distance': self.distance.raw_data - self.distance.raw_data[0],
             'altitude': self.altitude.raw_data - self.altitude.raw_data[0],
             'velocity': self.velocity.filtered
             }
        if d['time'][0] != 0:
            print d['time']

        return pd.DataFrame(d)

    def __repr__(self):
        return '<%s, %s>' % \
            (self.score, len(self.distance.raw_data))

class StravaStream(object):

    def __init__(self, stream_type, stream_dict, change_units=True):
        self.raw_data = np.array(stream_dict['data'])
        self.stream_type = stream_type
        self.filter()

    def filter(self):
        self.filtered = smooth(self.raw_data, 'scipy')

    def convert_units(self):
        if self.stream_type == 'distance':
            self.raw_data /= meters_per_mile
        elif self.stream_type == 'velocity':
            self.raw_data *= 3600
            self.raw_data /= meters_per_mile
        elif self.stream_type == 'altitude':
            self.raw_data *= feet_per_meter

        self.filter()

if __name__ == '__main__':
    pass