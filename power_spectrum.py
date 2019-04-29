import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.signal as signal
import scipy.fftpack as fp


def butter_filter(x, cutoff):
    filter_order = 2
    samples_per_s = 200
    B, A = signal.butter(filter_order, cutoff / (samples_per_s / 2), 'low')
    xs = signal.filtfilt(B, A, x)
    ys = signal.filtfilt(B, A, y)
    return xs, ys


def get_fft(w, sample_rate):
    n = len(w)
    k = np.arange(n)
    T = n / sample_rate
    frq = (k / T)[range(n // 2)]
    Y = abs(fp.fft(w)) / n
    Y = Y[range(n // 2)] 
    return frq, Y


def get_power_spectrum_v(xs, ys, ts, dtheta=0.005, tolerance=0.000001):
    xf = interpolate.UnivariateSpline(ts, xs, k=3, s=0)
    yf = interpolate.UnivariateSpline(ts, ys, k=3, s=0)
    tsn = np.arange(ts[0], ts[-1], 0.001)
    xv = xf.derivative(1)(tsn)
    yv = yf.derivative(1)(tsn)
    angle = np.unwrap(np.arctan2(yv, xv))
    anglef = interpolate.UnivariateSpline(tsn, angle, k=3, s=0)
    angledf = anglef.derivative(1)
    vel = np.sqrt(xv ** 2 + yv ** 2)
    vi = interpolate.UnivariateSpline(tsn, vel, k=3, s=0)
    pos_angle_d = abs(angledf(tsn))
    pos_angle_df = interpolate.UnivariateSpline(tsn, pos_angle_d, k=3, s=0)
    pos_angle = pos_angle_df.antiderivative(1)

    def newton_raphson():
        time_vals = []
        x = tsn[0]
        max_time = tsn[-1]
        for ta in np.arange(pos_angle(x), pos_angle(max_time), dtheta):
            for i in range(10):
                ax = pos_angle(x)
                if abs(ax - ta) <= tolerance:
                    break
                dfa = pos_angle_df(x)
                x = x + ta / dfa - ax / dfa
            time_vals.append(x)
        return time_vals

    t2 = newton_raphson()
    v_resampled = vi(t2)

    # plt.plot(ts, vel)
    # plt.plot(t2, v_resampled)
    # plt.show()

    logV0 = np.log(v_resampled)
    logV1 = logV0[~np.isnan(logV0)]
    logV = signal.detrend(logV1)
    freq, Y = get_fft(logV, (2 * np.pi) / dtheta)

    return freq, Y

def distance(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def triangle_area(a, b, c):
    s = (a + b + c) / 2
    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    return area

def get_power_spectrum_c(xs_raw, ys_raw, ts_raw, dtheta=0.005, tolerance=0.000001):    
    
    xf = interpolate.UnivariateSpline(ts_raw, xs_raw, k=3, s=0)
    yf = interpolate.UnivariateSpline(ts_raw, ys_raw, k=3, s=0)
    ts = np.arange(ts_raw[0], ts_raw[-1], 0.005)
    
    xs = xf(ts)
    ys = yf(ts)
    xv = xf.derivative(1)(ts)
    yv = yf.derivative(1)(ts)
    
    C = np.zeros(len(ts))
    for i in range(1, len(xs) - 1):
        a = distance(xs[i-1], ys[i-1], xs[i], ys[i])
        b = distance(xs[i+1], ys[i+1], xs[i], ys[i])
        c = distance(xs[i+1], ys[i+1], xs[i-1], ys[i-1])
        C[i] = (4 * triangle_area(a, b, c)) / (a * b * c)

    angle = np.unwrap(np.arctan2(yv, xv))
    anglef = interpolate.UnivariateSpline(ts, angle, k=3, s=0)
    angledf = anglef.derivative(1)
    pos_angle_d = abs(angledf(ts))
    pos_angle_df = interpolate.UnivariateSpline(ts, pos_angle_d, k=3, s=0)
    pos_angle = pos_angle_df.antiderivative(1)

    def newton_raphson():
        time_vals = []
        x = ts[0]
        max_time = ts[-1]
        for ta in np.arange(pos_angle(x), pos_angle(max_time), dtheta):
            for i in range(30):
                ax = pos_angle(x)
                if abs(ax - ta) <= tolerance:
                    break
                dfa = pos_angle_df(x)
                x = x + ta / dfa - ax / dfa
            time_vals.append(x)
        return time_vals
    
    cut = 10
    t2 = newton_raphson()[cut:-cut]
    
    curva = interpolate.UnivariateSpline(ts, C, k=3, s=0)
    curva_resampled = curva(t2)
       
    #plt.plot(t2, curva_resampled)
    #plt.show()
    logC0 = np.log(curva_resampled)
    logC1 = logC0[~np.isnan(logC0)]
    logC = signal.detrend(logC1)
    
    freq, Y = get_fft(logC, (2 * np.pi) / dtheta)

    return freq, Y
  
