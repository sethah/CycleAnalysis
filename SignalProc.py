import numpy as np
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
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
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y_new = np.concatenate((firstvals, y, lastvals))
    
    output = np.convolve(m[::-1], y_new, mode='valid')

    if output.shape[0] == y.shape[0]:
        print y.shape[0], output.shape[0]

    return output

def smooth(vec, copy=True):
    if copy:
        vec = np.copy(vec)

    return savitzky_golay(vec.ravel(), 51, 3)

def weighted_average(vals, weights):
    w = np.diff(weights)

    avg = np.dot(vals[0:vals.shape[0] - 1], w) / np.sum(w)
    if np.sum(w) == 0:
        print weights, w

    return avg

def diff(x, t):
    dxdt = np.diff(x, axis=0) / np.diff(t, axis=0).astype(float)
    dxdt_flat = np.append(dxdt, dxdt[-1])
    return dxdt_flat.reshape(dxdt.shape[0] + 1, dxdt.shape[1])

def main():
    pass

if __name__ == '__main__':
    main()