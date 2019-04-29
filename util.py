import numpy as np
import scipy.interpolate
import scipy.signal as signal
from collections import deque
import scipy.fftpack as fp
from numba import njit

def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    dif = a - b
    dif_squared = dif ** 2
    mean_of_dif = dif_squared.mean()
    rmse_val = np.sqrt(mean_of_dif)
    return rmse_val

def butter_filter(x, cutoff, samples_per_s=200, filter_order=2):
    xc = np.copy(x)
    B, A = signal.butter(filter_order, cutoff / (samples_per_s / 2), 'low')
    xs = signal.filtfilt(B, A, xc)
    return xs

def sm(x, cutoff=0.5, cut=500, samples_per_second=100):
    """ smoothing and cutting beginning and end """
    if cutoff > 0.0:
        x0 = butter_filter(x, cutoff, samples_per_second)
    else:
        x0 = x
    x1 = x0[cut: len(x0)-cut]
    return x1

def interpolate(ts_raw, xs, dt):
    ts = [t - ts_raw[0] for t in ts_raw]
    xc = np.copy(xs)
    x_spline = scipy.interpolate.UnivariateSpline(ts, xc, k=3, s=0)
    new_ts = np.arange(ts[0], ts[-1], dt)
    new_xs = x_spline(new_ts)
    return new_ts, new_xs

class empty(object):
    pass


class delay_line():
    def __init__(self, length, init_value=0):
        self.delay_line = deque([init_value] * length)

    def addpop(self, x):
        self.delay_line.appendleft(x)
        return self.delay_line.pop()


class control_system():
    def __init__(self, gain, delay, slowing, dt, init_qo = 0, init_p = 0):
        self.gain = gain
        self.delay = delay
        self.slowing = slowing
        self.dt = dt
        self.qo = init_qo
        self.input_line = delay_line(self.delay, init_p)

    def input_function(self, qi):
        p = self.input_line.addpop(qi)
        return p

    def step(self, qi, r):
        p = self.input_function(qi)
        e = r - p
        self.qo += (self.gain * e - self.qo) * (self.dt / self.slowing)
        return self.qo


def load_columnar(filename, dtype=np.float):
    fn = open(filename)
    names = fn.readline()[1:].split()
    data = np.loadtxt(filename, unpack=True, skiprows=1, dtype=dtype)
    d = dict(zip(names, data))
    fn.close()
    return d


def fft(w, sample_rate):
    n = len(w)
    k = np.arange(n)
    T = n / sample_rate
    frq = (k / T)[range(n // 2)]
    Y = abs(fp.fft(w)) / n
    Y = Y[range(n // 2)]
    return frq, Y


def rmsep(a, b):
    """ error in ratio of total range of a """
    a = np.asarray(a)
    b = np.asarray(b)
    dif = a - b
    dif_squared = dif ** 2
    mean_of_dif = dif_squared.mean()
    rmse_val = np.sqrt(mean_of_dif)

    rangea = np.max(a) - np.min(a)
    rmse_percent = rmse_val/rangea
    return rmse_percent

def rmse_percent(a, b):
    """ error in ratio of total range of a """
    ## assume numpy array
    dif = a - b
    dif_squared = dif ** 2
    mean_of_dif = dif_squared.mean()
    rmse_val = np.sqrt(mean_of_dif)
    rangea = np.max(a) - np.min(a)
    rmse_ratio = (rmse_val/rangea)
    rmse_percents = "{:.3f}%".format(100.0 * rmse_ratio)
    return rmse_percents

#rmse_percent a, b = (np.asarray([10,10]), np.asarray([5,5]))

def get_vel(ts, xs):
    return scipy.interpolate.UnivariateSpline(ts, xs, k=3, s=0).derivative(1)(ts)
