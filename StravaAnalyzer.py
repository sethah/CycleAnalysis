import numpy as np
import pandas as pd
import requests as r
import pymongo
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns
from math import factorial


class StravaAnalyzer(object):

    def __init__(self, query={}):
        self.client = pymongo.MongoClient()
        self.db = self.client.mydb
        self.table = self.db.activities

        result = self.table.find(query)
        activities = [activity for activity in result]
        self.stream_matrices(activities)

    def stream_matrices(self, activities):
        self.dist = self.construct_matrix(activities, 'distance')
        self.time = self.construct_matrix(activities, 'time')
        self.grade = self.construct_matrix(activities, 'grade_smooth')
        self.vel = self.construct_matrix(activities, 'velocity_smooth')
        self.watt = self.construct_matrix(activities, 'watts')
        self.alt = self.construct_matrix(activities, 'altitude')
        self.accel = self.diff(self.vel, self.time)

    def py_matrix(self, activities, metric):
        min_length = 990
        l = []
        for activity in activities:
            if metric not in activity['streams']:
                l.append([0]*min_length)
            else:
                if len(activity['streams'][metric]['data']) < min_length:
                    continue
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

    def diff(self, x, t):
        dxdt = np.diff(x, axis=0) / np.diff(t, axis=0).astype(float)
        dxdt_flat = np.append(dxdt, dxdt[-1])
        return dxdt_flat.reshape(dxdt.shape[0] + 1, dxdt.shape[1])

    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

    def hill_analysis(self, d_alt, idx):
        min_grade = 0
        time_dev_window = 45
        # grade = np.where(self.grade[:, idx] > 4)
        # print grade
        hills = []
        filtered_grade = np.copy(self.savitzky_golay(self.grade[:, idx], 51, 3))
        # print filtered_grade[k]
        hill_start = False
        hill_start_idx = 0
        for k in xrange(self.dist.shape[0]):
            time_elapsed = self.time[k][idx] - self.time[hill_start_idx][idx]
            if d_alt[k] > min_grade and not hill_start:
                hill_start = True
                hill_start_idx = k
                # print filtered_grade[k]

            elif hill_start and (d_alt[k] < 0 or k == self.dist.shape[0] - 1):
                hill_start = False
                distance = self.dist[k][idx] - self.dist[hill_start_idx][idx]
                if distance < 5:
                    continue

                grade = 3
                score = distance*grade
                hill = {}
                # hill['avg_grade'] = np.mean()
                hill['index'] = (hill_start_idx, k)
                hill['distance'] = (self.dist[hill_start_idx][idx], self.dist[k][idx])
                hills.append(hill)

        return hills

def rect(x, y, w, h, c, ax):
    # ax = plt.gca()
    polygon = plt.Rectangle((x, y), w, h, color=c)
    ax.add_patch(polygon)

if __name__ == '__main__':
    s = StravaAnalyzer()
    # print s.dist.shape
    

    fig, axs = plt.subplots(1,1)
    ax = axs
    ax.plot(s.dist[:, 3], s.alt[:, 3])
    ax1 = ax.twinx()
    # ax1.plot(s.dist[:, 3],
    #              s.savitzky_golay(s.grade[:, 3], 51, 3),
    #              c='r', label=str(3))
    d_alt = s.diff(s.alt[:,3][:, np.newaxis], s.dist[:, 3][:, np.newaxis])
    print d_alt.shape
    ax1.plot(s.dist[:, 3], s.savitzky_golay(d_alt.ravel(),51,3), c='g')
    ax1.grid(b=False)
    hills = s.hill_analysis(s.savitzky_golay(d_alt.ravel(),51,3), 3)
    for hill in hills:
        color = 'r'
        for sample in xrange(hill['index'][0], hill['index'][1]):
            w = s.dist[sample+1,3] - s.dist[sample,3]
            rect(s.dist[sample,3], 0, w, s.alt[sample,3], color, ax)
            

    plt.show()

    # fig, axs = plt.subplots(5, 3)

    # for idx, ax in enumerate(axs.reshape(-1)):
    #     ax.plot(s.dist[:, idx], s.alt[:, idx])
    #     ax1 = ax.twinx()
    #     ax1.plot(s.dist[:, idx],
    #              s.savitzky_golay(s.grade[:, idx], 51, 3),
    #              c='r', label=str(idx))
    #     ax1.grid(b=False)
    #     ax1.legend()

    # plt.show()