#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import datetime as dt

"""
Created on Fri Dec  4 12:32:27 2020

@author: philipp
"""

def average_datasets_mult(data_average, key, time_init, time_end, delta, time_key='date time'):
    # Find corresponding entries and average
    # Criterion: +- delta seconds. Averaging all datapoints inbetween 
    idx = []
    date = []
    time_now = time_init
    while time_now < time_end:
        idx_0 = np.where(data_average[time_key] > time_now-dt.timedelta(minutes=delta))
        idx_1 = np.where(data_average[time_key] < time_now)#+dt.timedelta(minutes=delta))
        intersect = np.intersect1d(idx_0, idx_1)
        if len(intersect) > 0:
            idx.append(intersect)
            date.append(time_now)
        time_now += dt.timedelta(minutes=delta)
        
    mean_key  = [[] for ii in range(len(key))]
    mean_date = []
    num = []
    for ii in range(len(idx)):
        if date[ii] not in mean_date:
            num.append(idx[ii].size)
            mean_date.append(date[ii])
            for kk in range(len(key)):
                try:
                    mean_key[kk].append(np.mean(np.array(data_average[key[kk]].iloc[idx[ii]], dtype=np.float64)))
                except ValueError:
                    print(data_average[key[kk]].iloc[idx[ii]])
    out_dict = {time_key : mean_date, 'datapoints' : num}
    for kk in range(len(key)):
        out_dict.update({key[kk] : mean_key[kk]})
    mean_corresponding = pd.DataFrame(out_dict) 
                
    return mean_corresponding

def average(fname_int, delta_t, column_date, fname_out):
    
    results = pd.read_csv(fname_in, parse_dates=[column_date])

    ## Discard invalid results CNET
    filter_invalid = True
    cwp_min = 0.0
    idx_valid = np.array([])
    for ii in range(len(results)):
        if results['down_flux_all_lw(Wm-2)'].iloc[ii] == 0.0:
            continue
        elif filter_invalid and (np.abs(results['diff_lwp(gm-2)'].iloc[ii]) > 1.0 and \
             np.abs(results['diff_lwp(gm-2)'].iloc[ii]) < 2000.0):
            continue
        elif np.float(results['cwp(gm-2)'].iloc[ii]) < cwp_min:
            continue
        else:
            idx_valid = np.concatenate((idx_valid, [ii]))
        
    results  =results.iloc[idx_valid]
    idx_arctic = np.where(results['time'] >= dt.datetime(2017, 5, 30))[0]
    results = results.iloc[idx_arctic]

    
    #delta_t = 60.
    keys = list(results.keys())
    idx_keys = np.where(np.array(keys) == 'time')[0][0]
    keys.pop(idx_keys)

    if delta_t > 0.:
        results = average_datasets_mult(results, keys, \
                                      time_init=dt.datetime(2017, 5, 30), \
                                      time_end=dt.datetime(2017, 7, 19), delta=delta_t, \
                                      time_key='time')
    if filter_invalid:
        results.to_csv("{}_AVG_{}_FILTER.csv".format(fname_out, int(delta_t)), index=False)
    else:
        results.to_csv("{}_AVG_{}.csv".format(fname_out, int(delta_t)))

if __name__ == '__main__':
    fname_in = sys.argv[1]
    delta_t = float(sys.argv[2])
    column_date = int(sys.argv[3])#18
    fname_out = sys.argv[4]
    average(fname_in, delta_t, column_date, fname_out)