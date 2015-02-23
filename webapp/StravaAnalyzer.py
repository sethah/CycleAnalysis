import numpy as np
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns
from datetime import datetime
from SignalProc import weighted_average, smooth, diff
from PlotTools import PlotTool
import brewer2mpl
cmap = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors


class StravaAnalyzer(object):

    def __init__(self, query={}):
        # some constants
        self.meters_per_mile = 1609.34
        self.feet_per_meter = 3.280

        # db params
        self.client = pymongo.MongoClient()
        self.db = self.client.mydb
        self.table = self.db.activities

        # structure the data
        self.get_activities()
        self.stream_matrices(self.activities)

        self.df = None
        

    def get_activities(self, query={}):
        min_length = 990
        result = self.table.find(query)
        activities = [activity for activity in result]
        print len(activities)
        # self.activities = [activity for activity in activities\
        #     if len(activity['streams']['distance']['data']) > min_length]
        # activities = []
        self.activities = []
        for activity in activities:
            if self.is_ride(activity) and len(activity['streams']['distance']['data']) > min_length:
                self.activities.append(activity)
        # self.activities = activities

    def is_ride(self, activity):
        min_vel = 8. / 3600 * self.meters_per_mile
        max_vel = 30. / 3600 * self.meters_per_mile
        # vel = activity['streams']['velocity_smooth']['data']
        avg_vel = np.mean(activity['streams']['velocity_smooth']['data'])
        # print avg_vel

        return avg_vel >= min_vel and avg_vel <= max_vel

    def stream_matrices(self, activities):
        self.dist = self.construct_matrix(activities, 'distance')
        self.time = self.construct_matrix(activities, 'time')
        self.grade = self.construct_matrix(activities, 'grade_smooth')
        self.vel = self.construct_matrix(activities, 'velocity_smooth')
        self.watt = self.construct_matrix(activities, 'watts')
        self.alt = self.construct_matrix(activities, 'altitude')
        self.accel = diff(self.vel, self.time)

    def py_matrix(self, activities, metric):
        min_length = 990
        l = []
        for idx, activity in enumerate(activities):
            if metric not in activity['streams']:
                l.append([0]*min_length)
            else:
                l.append(activity['streams'][metric]['data'])

        min_length = min(map(len, l))
        l = [item[:min_length] for item in l]
        
        return l

    def construct_matrix(self, activities, metric):
        X = np.array(self.py_matrix(activities, metric)).T
        X = self.pre_proc(X, metric)

        return X

    def pre_proc(self, mat, metric):
        if metric == 'distance':
            meters_per_mile = 1609.34
            mat /= meters_per_mile
            for idx in xrange(mat.shape[1]):
                dist_gap = mat[0][idx] - mat[0][0]
                mat[:,idx] -= dist_gap
        elif metric == 'time':
            for idx in xrange(mat.shape[1]):
                mat[:,idx] -= mat[0][idx]
        elif metric == 'accel':
            for idx in xrange(mat.shape[1]):
                mat[:,idx] = diff(mat[:,idx])
        else:
            mat = mat

        return mat

    def strava_date(date_string):
        s = date_string.split('T')[0]
        return datetime.strptime(date_string, '%Y-%m-%d')

    def hill_score(self, ride_idx, hill_indices):
        start = hill_indices[0]
        stop = hill_indices[1]
        grade_vec = self.grade[start:stop, ride_idx]
        dist_vec = self.dist[start:stop, ride_idx]
        mean_grade = weighted_average(grade_vec, dist_vec)
        distance = distance = self.dist[stop][ride_idx] - self.dist[start][ride_idx]
        
        return distance*mean_grade*self.meters_per_mile

    def merge_hills(self, hill1, hill2):
        hill = self.construct_hill(hill1['ride_idx'], hill1['start'], hill2['stop'], hill1['previous_climb'])
        return hill

    def clean_hills(self, hills):
        min_score = 8000
        min_separation = 0.5
        clean_hills = []
        for k, hill in enumerate(hills):
            if len(clean_hills) != 0:
                if hill['distance'][0] - clean_hills[-1]['distance'][1] < min_separation:
                    hill = self.merge_hills(clean_hills[-1], hill)
                    clean_hills.pop()
            if hill['score'] > min_score:
                clean_hills.append(hill)

        return clean_hills

    def construct_hill(self, ride_idx, start, stop, climb_total, previous_hill=None):
        hill = {}
        mean_grade = weighted_average(self.grade[start:stop + 1, ride_idx],
                                                   self.dist[start:stop + 1, ride_idx])
        mean_velocity = weighted_average(self.vel[start:stop + 1, ride_idx],
                                              self.dist[start:stop + 1, ride_idx])
        hill['ride_idx'] = ride_idx
        hill['velocity'] = mean_velocity / self.meters_per_mile * 3600
        hill['previous_distance'] = self.dist[start][ride_idx]
        hill['previous_climb'] = climb_total
        hill['start'] = start
        hill['stop'] = stop
        hill['time_riding'] = self.time[start][ride_idx] - self.time[0][ride_idx]
        hill['distance'] = (self.dist[start][ride_idx], self.dist[stop][ride_idx])
        hill['score'] = self.hill_score(ride_idx, (start, stop))
        hill['activity_id'] = self.activities[ride_idx]['id']
        
        return hill

    def hill_analysis(self, idx):
        d_alt = smooth(diff(self.alt[:,idx][:, np.newaxis],
                         self.dist[:, idx][:, np.newaxis]))

        hills = []
        hill_start = False
        hill_start_idx = 0
        climb_total = np.zeros(self.dist.shape[0])
        for k in xrange(self.dist.shape[0]):
            time_elapsed = self.time[k][idx] - self.time[hill_start_idx][idx]
            if k != 0:
                climb_total[k] = climb_total[k - 1] + \
                             np.max([self.alt[k][idx] - self.alt[k - 1][idx], 0])

            if d_alt[k] > 0 and not hill_start:
                hill_start = True
                hill_start_idx = k

            elif hill_start and (d_alt[k] < 0 or k == self.dist.shape[0] - 1):
                hill_start = False
                hill = self.construct_hill(idx, hill_start_idx, k, climb_total[hill_start_idx])
                hills.append(hill)

        return self.clean_hills(hills)

    def get_hill_features(self, hill, ride_idx):
        activity_id = self.activities[ride_idx]['id']

        # get raw vectors for the hill
        hill_alt = self.alt[hill['start']:hill['stop'], ride_idx]
        hill_time = self.time[hill['start']:hill['stop'], ride_idx]
        hill_distance = self.dist[hill['start']:hill['stop'], ride_idx]

        # start from zero
        time_elapsed = (hill_time - hill_time[0])
        distance_elapsed = (hill_distance - hill_distance[0])
        climb_elapsed = (hill_alt - hill_alt[0])

        # filter the grade and velocity
        grade_smooth = smooth(self.grade[:, ride_idx])[hill['start']:hill['stop']]
        velocity_smooth = smooth(self.vel[:, ride_idx])[hill['start']:hill['stop']]

        df = pd.DataFrame({'hill_grade_smooth': grade_smooth,
                          'time_elapsed': time_elapsed,
                          'distance_elapsed': distance_elapsed,
                          'climb_elapsed': climb_elapsed,
                          'velocity_smooth': velocity_smooth,
                          'activity_id': [activity_id]*len(grade_smooth),
                          'start': [hill['start']]*len(grade_smooth)},
                         index=np.arange(grade_smooth.shape[0]))
        

        return df

    def hills_df(self, ride_idx):
        hill_summary_df = self.hill_summary()
        hills = self.hill_analysis(ride_idx)
        
        for i, hill in enumerate(hills):
            hill_df = self.get_hill_features(hill, ride_idx)
            if i == 0:
                hills_df = hill_df
            else:
                hills_df = hills_df.append(hill_df, ignore_index=True)

        hills_df = pd.merge(hills_df, hill_summary_df, on=['activity_id', 'start'], suffixes=['','_'])
        return hills_df, hill_summary_df

    def hill_summary(self):
        df = pd.read_csv('hills.csv')
        df = df.drop(['Unnamed: 0', 'distance'], 1)

        df = pd.merge(df, df[['stop', 'activity_id']].shift(1),
                      left_index=True, right_index=True, suffixes=['', '_prev'])

        # get the stop index of the previous hill
        df['stop_prev'] = np.where(df['activity_id_prev'] == df['activity_id'], df['stop_prev'], np.nan)
        df['stop_prev'] = df['stop_prev'].fillna(0)
        
        # get attributes of the last climb
        df['time_since_last_climb'] = self.time[df['stop_prev'], df['ride_idx']]
        df['score_last_climb'] = np.where(df['stop_prev'] != 0, df['score'].shift(1), 0)
        
        return df

    def fill_df(self):
        for idx in xrange(self.dist.shape[1]):
            hills = self.hill_analysis(idx)
            if len(hills) != 0:
                if self.df is None:
                    self.df = pd.DataFrame(hills)
                else:
                    self.df = self.df.append(pd.DataFrame(hills))

        self.df.to_csv('hills.csv')


    def plot_activities(self, indices):
        hills_list = []
        for activity in xrange(indices[0], indices[1]):
            hills = self.hill_analysis(activity)
            if len(hills) != 0:
                hills_list.append((activity, hills))


        r, c = self.subplot_dims(len(hills_list))
        fig, axs = plt.subplots(r, c, figsize=(15,12))

        for k, ax in enumerate(axs.reshape(-1)):
            # hills = self.hill_analysis(ride_idx)
            self.plot_hills(hills_list[k][0], ax, hills_list[k][1])
            ax.legend()

def rect(x, y, w, h, c, ax):
    polygon = plt.Rectangle((x, y), w, h, color=c)
    ax.add_patch(polygon)

if __name__ == '__main__':
    s = StravaAnalyzer()
    # s.fill_df()
    # hills_df = s.hills_df(2)
    # hills_df.head()
    print s.activities[0].keys()