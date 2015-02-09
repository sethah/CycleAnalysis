import numpy as np
import pandas as pd
import requests as r
import matplotlib.pyplot as plt
import pymongo
import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns
import scipy.signal as scs

ACCESS_TOKEN = '535848c783c374e8f33549d22f089c1ce0d56cd6'
def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

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

def get_efforts(segment_id):
    url = 'https://www.strava.com/api/v3/segments/%s/leaderboard' % segment_id
    payload = {'per_page': 10}
    header = {'Authorization': 'Bearer %s' % ACCESS_TOKEN}
    response = r.get(url, headers=header, params=payload)
    print response.content
    data = response.json()
    effort_ids = [(entry['effort_id'], entry['athlete_name']) for entry in data['entries']]

    efforts = []
    for eid in effort_ids:
        efforts.append(get_effort(eid[0], eid[1]))
    
    return efforts


def get_effort(effort_id, name, types=None):
    if types is None:
        types = ['time','latlng','distance','altitude',
                 'watts', 'velocity_smooth', 'moving', 'grade_smooth']
    payload = {'resolution': 'medium'}
    url = 'https://www.strava.com/api/v3/segment_efforts/%s/streams/%s' % (effort_id, ','.join(types))
    header = {'Authorization': 'Bearer %s' % ACCESS_TOKEN}
    response = r.get(url, headers=header, params=payload)
    data = response.json()
    data = {x['type']:x for x in data}
    data['name'] = name

    return data

def store_efforts(segment_id):
    client = pymongo.MongoClient()
    db = client.mydb
    table = db.efforts
    efforts = get_efforts(segment_id)

    for effort in efforts:
        effort['segment_id'] = segment_id
        if table.find({'segment_id': segment_id, 'name': effort['name']}).count() != 0:
            print 'Duplicate! ', effort['name']
            continue
        table.insert(effort)

def pre_proc(mat, metric):
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


def remove_short_efforts(efforts, min_length=995):
    efforts = [effort for effort in efforts if len(effort['distance']['data']) >= min_length]
    # print len(efforts)
    return efforts


def py_matrix(efforts, metric):
    min_length = 995
    l = []
    for effort in efforts:
        if metric not in effort:
            l.append([0]*min_length)
        else:
            l.append(effort[metric]['data'])
    # l = [effort[metric]['data'] for effort in efforts if len(effort[metric]['data']) > min_length]
    min_length = min(map(len, l))
    l = [item[:min_length] for item in l]
    
    return l


def construct_matrix(efforts, metric):
    X = np.array(py_matrix(efforts, metric)).T
    X = pre_proc(X, metric)

    return X

def plot_fill(X, Y, Z, cmap, ax):
    ax.plot(X, Y)

    dx = X[1] - X[0]
    N = X.shape[0]

    c = np.arange(X.shape[0])
    Zsort = np.argsort(Z)
    Z[Zsort] = c
    # print Zsort

    for n, (x, y, z) in enumerate(zip(X, Y, Z)):
        color = cmap(z / float(N))
        # print color
        rect(x, 0, dx, y, color, ax)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def rect(x, y, w, h, c, ax):
    # ax = plt.gca()
    polygon = plt.Rectangle((x, y), w, h, color=c)
    ax.add_patch(polygon)


def diff(x, t):
    dxdt = np.diff(x, axis=0) / np.diff(t, axis=0).astype(float)
    dxdt_flat = np.append(dxdt, dxdt[-1])
    return dxdt_flat.reshape(dxdt.shape[0] + 1, dxdt.shape[1])


def get_intervals(y, t, n=4, min_duration=60):

    ysort = sorted(y, reverse=True)
    percentile = 0.95
    while percentile > 0.7:
        thresh = ysort[len(y) - int(len(y)*percentile)]
        prev = 0
        intervals = []
        cross_high, cross_low = None, None
        for idx, sample in enumerate(y):
            if sample >= thresh and prev < thresh:
                # print 'high', sample, thresh, idx
                cross_high = idx
            elif sample < thresh and prev >= thresh:
                # print 'low', sample, thresh, idx
                cross_low = idx
            elif idx == 0 and sample >= thresh:
                # first sample was in the high category
                cross_high = idx
            elif idx == len(y) - 1 and sample >= thresh:
                # last sample was high
                cross_low = idx
            else:
                prev = sample
                continue

            if cross_high is not None and cross_low is not None:
                interval = (cross_high, cross_low)
                cross_high, cross_low = None, None
                if check_interval(t, interval, min_duration):
                    intervals.append(interval)

            prev = sample
        
        if len(intervals) >= n:
            break
        percentile -= 0.05

    # print percentile
    return intervals, percentile

def check_interval(t, interval, min_duration):
    # print 'interval length: ', t[interval[1]] - t[interval[0]]
    return (t[interval[1]] - t[interval[0]] > min_duration)
def main():

    segment_ids = [7673423, 4980024]
    for segment_id in segment_ids:
        store_efforts(segment_id)
if __name__ == '__main__':
    # main()


    client = pymongo.MongoClient()
    db = client.mydb
    table = db.efforts

    segment_ids = [7673423, 4980024]
    segment_id = segment_ids[1]
    efforts = table.find({'segment_id': segment_id})

    # for e in efforts:
    #     print e['name']

    efforts = table.find({'segment_id': segment_id})
    # print efforts.count()
    # # # # print efforts.count()
    the_efforts = [effort for effort in efforts]
    the_efforts = remove_short_efforts(the_efforts)

    names = [effort['name'] for effort in the_efforts]
    print names
    X_dist = construct_matrix(the_efforts, 'distance')
    X_time = construct_matrix(the_efforts, 'time')
    X_grade = construct_matrix(the_efforts, 'grade_smooth')
    X_vel = construct_matrix(the_efforts, 'velocity_smooth')
    X_watt = construct_matrix(the_efforts, 'watts')
    X_alt = construct_matrix(the_efforts, 'altitude')
    X_accel = diff(X_vel, X_time)

    cmap = get_cmap(X_alt.shape[0])
    cmap=plt.get_cmap("jet")

    my_idx = names.index('Seth Hendrickson')

    fig, axs = plt.subplots(3, 2, figsize=(15,10))
    grade_filt = savitzky_golay(X_grade[:, my_idx], 51, 3)
    vel_pct_filt = savitzky_golay(X_vel[:, my_idx] / X_vel[:, 0], 51, 3)
    accel_filt = savitzky_golay(X_accel[:, 0], 51, 3)
    # plot_fill(X_dist[:, 0], X_alt[:, 0], grade_filt, cmap, axs[0])
    # plot_fill(X_dist[:, 0], X_alt[:, 0], np.copy(vel_pct_filt), cmap, axs[1])
    # plot_fill(X_dist[:, 0], X_alt[:, 0], accel_filt, cmap, axs[2])
    start = 700
    for k, ax in enumerate(axs.reshape(-1)):
        vel_pct_filt = savitzky_golay(X_vel[:, my_idx] / X_vel[:, k], 51, 3)
        plot_fill(X_dist[start:995, k], X_alt[start:995, k], np.copy(vel_pct_filt[start:995]), cmap, ax)
        ax2 = ax.twinx()
        # ax2.plot(X_dist[:, k], vel_pct_filt, label=names[k])
        ax2.plot(X_dist[start:995, k], savitzky_golay(X_grade[start:995, k], 51, 3), label=names[k])
        ax2.grid(b=False)
        ax2.legend()

    # # plt.legend()
    # ax2 = axs[0].twinx()
    # ax2.plot(X_dist[:, 0], grade_filt)
    # ax2.grid(b=False)
    
    # xmin, xmax = np.min(X_dist[:, 0]), np.max(X_dist[:, 0])
    # y = vel_pct_filt
    # t = X_time[:, 4]
    # intervals, percentile = get_intervals(y, t)
    # for interval in intervals:
    #     low = interval[0]
    #     high = interval[1]
    #     print X_dist[low][4], X_dist[high][4]

    # y.sort()
    # ytop = y[::-1][len(y) - int(len(y)*percentile)]

    # ax3.plot([xmin, xmax],[ytop, ytop])
    # ax3 = axs[2].twinx()
    # ax3.plot(X_dist[:, 0], accel_filt)
    # ax3.grid(b=False)
    plt.show()
    



    

