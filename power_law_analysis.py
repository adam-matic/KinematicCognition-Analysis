import os
import json
import math
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import scipy.interpolate as interpolate


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    dif = a - b
    dif_squared = dif ** 2
    mean_of_dif = dif_squared.mean()
    rmse_val = np.sqrt(mean_of_dif)
    return rmse_val


def load_points(data):
    if isinstance(data, list):
        xs_raw, ys_raw, ts_raw = data
    elif isinstance(data, dict):
        xs_raw = data['cx']
        ys_raw = data['cy']
        ts_raw = data['ts']
    elif isinstance(data, str):
        return load_points(json.load(open(data)))
    else:
        print('error in input type')
        return
    return xs_raw, ys_raw, ts_raw


def butter_filter(x, y, cutoff, samples_per_s=200, filter_order = 2):
    B, A = signal.butter(filter_order, cutoff / (samples_per_s / 2), 'low')
    xs = signal.filtfilt(B, A, x)
    ys = signal.filtfilt(B, A, y)
    return xs, ys


def cut_start_end(x, cut_start, cut_end):
    return x[(int(cut_start * 200)): len(x) - int(cut_end * 200)]

def get_beta(x, y, t):
    xspl = interpolate.UnivariateSpline(t, x, k=3, s=0)
    yspl = interpolate.UnivariateSpline(t, y, k=3, s=0)
    xs = xspl(t)
    ys = yspl(t)
    ts = np.copy(t)
    xvel = xspl.derivative(1)(ts)
    yvel = yspl.derivative(1)(ts)
    xacc = xspl.derivative(2)(ts)
    yacc = yspl.derivative(2)(ts)
    xjerk = xspl.derivative(3)(ts)
    yjerk = yspl.derivative(3)(ts)

    vel = np.sqrt(xvel**2.0 + yvel**2.0)
    D0 = np.abs( yacc * xvel - xacc * yvel)

    D = [np.NaN if a == 0.0 else a for a in D0]

    RC = (vel**3.0) / D

    C = 1.0 / RC
    A = vel / RC

    C_clean = C[np.isfinite(C)]
    A_clean = A[np.isfinite(A)]

    logC = np.log10(C_clean)
    logA = np.log10(A_clean)
    
    V_clean = vel[np.isfinite(vel)]
    R_clean = RC[np.isfinite(RC)]
    logV = np.log10(V_clean)
    logR = np.log10(R_clean)
    
    v_mean = np.mean(vel[np.isfinite(vel)])
    A_mean = np.mean(A_clean)

   
    beta, off, r, p_v, std_err = stats.linregress(logC, logA)
    r2 = r * r

    betaRV, offRV, rr, p_vs, std_ersr = stats.linregress(logR, logV)
    r2RV = rr * rr
    
    betaCV, offCV, rcv, p_v, std_err = stats.linregress(logC, logV)
    r2CV = rcv * rcv

    return {'x': xs,
            'y': ys,
            't': ts,
            'beta': beta,
            'offset': off,
            'r2': r2,
            'vel': vel,
            'Avel': A,
            'C': C,
            'RC': RC,
            'C_clean': C_clean,
            'A_clean': A_clean,
            'v_mean': v_mean,
            'A_mean': A_mean,
            'D': D,
            'logC': logC,
            'logA': logA,
            'logR': logR,
            'logV': logV,            
            'betaVR':betaRV,
            'r2VR': r2RV,
            'offsetVR': offRV,

            'betaCV':betaCV,
            'r2CV': r2CV,
            'offsetCV': offCV,
            
            'xvel' : xvel,
            'yvel' : yvel,
            'xacc' : xacc,
            'yacc' : yacc,
            'xjerk': xjerk,
            'yjerk': yjerk
            }


def analyze(data, butter='none', cut=3):
    if isinstance(data, list):
        xs_raw, ys_raw, ts_raw = data[0][:], data[1][:], data[2][:]
    elif isinstance(data, dict):
        xs_raw = data['xs'][:]
        ys_raw = data['ys'][:]
        ts_raw = data['ts'][:]
    elif isinstance(data, str):
        return analyze(json.load(open(data)),
                       butter=butter)
    else:
        print('error in input type')
        return

    ts_raw = [x - ts_raw[0] for x in ts_raw]

    xs_raw_spline = interpolate.UnivariateSpline(ts_raw, xs_raw, k=3, s=0)
    ys_raw_spline = interpolate.UnivariateSpline(ts_raw, ys_raw, k=3, s=0)

    ts = np.arange(ts_raw[0], ts_raw[-1], 0.005)
    #ts = np.arange(ts_raw[0], ts_raw[-1], 0.0166666)
    xs_spl = xs_raw_spline(ts)
    ys_spl = ys_raw_spline(ts)

    if butter == "None" or butter == "none" or not butter:
        d = get_beta(xs_spl, ys_spl, ts)
    else:
        xs, ys = butter_filter(xs_spl, ys_spl, butter)
        if (cut != 0):
            N = int(cut * 200)
            xs = xs[N:-N]
            ys = ys[N:-N]
            ts = ts[: -(2*N) ]
        d = get_beta(xs, ys, ts)

    return d

