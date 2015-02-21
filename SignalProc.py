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
    
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y_new = np.concatenate((firstvals, y, lastvals))
    
    output = np.convolve(m[::-1], y_new, mode='valid')

    # if output.shape[0] == y.shape[0]:
    #     print y.shape[0], output.shape[0]

    return output


def scipy_smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        # print "Input vector needs to be bigger than window size."
        return x

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d').astype(float)
    else:
        w = eval('np.' + window + '(window_len)').astype(float)

    y = np.convolve(w / w.sum(), s, mode='valid')
    output = y[(window_len/2 - 1):-(window_len/2)]

    if output.shape[0] != x.shape[0]:
        # print x.shape[0], output.shape[0]
        output = output[:-1]

    return output


def smooth(vec, filt_type, window_len=11, copy=True):
    if copy:
        vec = np.copy(vec)

    if filt_type == 'savgol':
        smoothed = savitzky_golay(vec.ravel(), 51, 3)
    elif filt_type == 'scipy':
        smoothed = scipy_smooth(vec.ravel(), window_len=window_len)

    return smoothed


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