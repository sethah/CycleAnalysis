import numpy as np
import pandas as pd
from SignalProc import weighted_average, smooth, diff
from datetime import datetime
import matplotlib.pyplot as plt
from PlotTools import PlotTool
import seaborn as sns

# global vars
meters_per_mile = 1609.34
feet_per_meter = 3.280

class StravaEffort(object):

    def __init__(self, effort_dict):
        self.init(effort_dict)

    def init(self, effort_dict):
        self.id = effort_dict['id']
        self.name = effort_dict['name']
        self.athlete = effort_dict['athlete']#['id']
        self.date = self.strava_date(effort_dict['start_date'])
        stream_dict = effort_dict['streams']
        self.velocity = self.init_stream(stream_dict, 'velocity_smooth')
        self.distance = self.init_stream(stream_dict, 'distance')
        self.time = self.init_stream(stream_dict, 'time')
        self.grade = self.init_stream(stream_dict, 'grade_smooth')
        self.altitude = self.init_stream(stream_dict, 'altitude')

    def init_stream(self, stream_dict, stream_type):
        return StravaStream(stream_type, stream_dict[stream_type])

    def stream_types(self):
        return {'velocity_smooth', 'distance', 'time', 'grade_smooth', 'altitude'}

    def strava_date(self, date_string):
        if type(date_string) != unicode:
            return date_string
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')


class StravaActivity(StravaEffort):

    def __init__(self, activity_dict):
        self.init(activity_dict)
        self.distance.convert_units()
        self.velocity.convert_units()
        self.altitude.convert_units()
        self.city = activity_dict['location_city']
        self.total_distance = activity_dict['distance']
        self.hills = self.hill_analysis()
        self.is_valid_ride = self.is_ride()

    def init_stream(self, stream_dict, stream_type):
        return StravaStream(stream_type, stream_dict[stream_type])

    def is_ride(self):
        min_vel = 4.  # mph
        max_vel = 30.  # mph

        avg_vel = np.mean(self.velocity.raw_data)
        return avg_vel >= min_vel and avg_vel <= max_vel

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
                                   'start_date': self.date,
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
        ax.set_title(self.date.date())
        axis_label_font = 20
        ax.set_xlabel('Distance (miles)', fontsize=axis_label_font)
        ax.set_ylabel('Altitude (ft)', fontsize=axis_label_font)
        ax.set_title('%s, %s, %s' % \
                        (self.name, self.athlete['id'], self.date.date()),
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
        fig.suptitle('Hills for %s on %s' % (self.name, self.date.date()),
                     fontsize=20)
        plt.show()

    def plot_filter(self, key):
        stream = getattr(self, key)
        x = smooth(stream.raw_data, 'scipy')
        print x.shape, self.distance.raw_data.shape
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

    def __repr__(self):
        return '<%s, %s, %s, %s>' % \
            (self.name, self.date, self.city, len(self.distance.raw_data))


class StravaHill(StravaEffort):

    def __init__(self, effort_dict, activity, previous_hill=None):
        self.init(effort_dict)
        self.activity = activity
        self.previous_hill = previous_hill
        self.score = self.hill_score()

    def hill_score(self):
        grade_vec = self.grade.raw_data
        dist_vec = self.distance.raw_data

        mean_grade = weighted_average(grade_vec, dist_vec)
        distance = self.distance.raw_data[-1] - self.distance.raw_data[0]

        return distance*mean_grade

    def hill_features(self):
        self.previous_climb = self.past_climb()
        # self.time_elapsed = self.time.raw_data[0]
        # self.distance_elapsed = self.distance.raw_data[0]
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
             'time_since_last_climb': [self.time_since_last_climb]*n,
             'score_last_climb': [self.score_last_climb]*n,
             'grade': self.grade.filtered,
             'time': self.time.raw_data - self.time.raw_data[0],
             'distance': self.distance.raw_data - self.distance.raw_data[0],
             'altitude': self.altitude.raw_data - self.altitude.raw_data[0],
             'velocity': self.velocity.filtered
             }
        # for key in d:
        #     print key, len(d[key])

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

if __name__ == '__main__':
    pass