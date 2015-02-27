import numpy as np
import pandas as pd
from SignalProc import weighted_average, smooth, diff
from StravaUser import StravaUser
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

class StravaModel(object):

    def __init__(self, model):
        self.model = model

    def predict(self, X, cols, window_len=6):
        pred = np.zeros(X.shape[0])
        prev_vel = np.ones(window_len)
        replace_indices = [i for i, col in enumerate(cols) if 'velocity' in col]
        dist_idx = np.where(cols == 'dist_int')[0]
        for i in xrange(X.shape[0]):
            row = X[i, :]
            for idx, j in enumerate(replace_indices):
                row[j] = prev_vel[idx]
            time_int = self.model.predict(row[np.newaxis, :])[0]
            vel = row[dist_idx] / (time_int + 1e-10)
            prev_vel = np.roll(prev_vel, 1)
            prev_vel[0] = vel
            pred[i] = time_int
        return pred

    def predict_activity(self, a):
        df = a.make_df()
        # df.pop('velocity')
        y = df.pop('time_int')
        X = df.values
        pred = self.predict(df.values, df.columns.values)
        pred_time = np.sum(pred)
        return pred, y, pred_time

    def streaming_predict(self, a, spacing=100):
        predicted_time = []
        dist = []
        df = a.make_df()
        # df.pop('velocity')
        df.pop('time_int')
        cols = df.columns.values
        for i in xrange(0, df.shape[0] - 1, spacing):
            X = df.iloc[i:,:].values
            pred = self.predict(X, cols)
            predicted_time.append(np.sum(pred) + (a.time.raw_data[i] - a.time.raw_data[0]))
            dist.append(a.distance.raw_data[i])
        return predicted_time, dist

if __name__ == '__main__':
    u = StravaUser('Seth')
    wlen = 2
    df = u.make_df((0, 30))
    # df.pop('velocity')
    y = df.pop('time_int')
    X = df.values
    model = RandomForestRegressor(max_depth=8)
    model.fit(X, y)
    m = StravaModel(model)
