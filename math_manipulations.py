# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:08:10 2020

@author: Philipp
"""

import datetime as dt
import numpy as np
import pandas as pd
import interpolate_era5_v2
from scipy.interpolate import interp1d

def parse_date_and_sza(file_):
    lat = np.float(file_.split("_")[3])
    lon = np.float(file_.split("_")[4])
    sza = np.float("{}.{}".format(file_.split("_")[5].split(".")[0], file_.split("_")[5].split(".")[1]))
    year = np.int(file_.split("_")[1][:4])
    month = np.int(file_.split("_")[1][4:6])
    day = np.int(file_.split("_")[1][6:])
    hour = np.int(file_.split("_")[2][:2])
    minute = np.int(file_.split("_")[2][2:4])
    sec = np.int(file_.split("_")[2][4:])  
    
    return lat, lon, sza, year, month, day, hour, minute, sec

def read_trace_gases(co2_path, co2_vmr, n2o_path, n2o_vmr, ch4_path, ch4_vmr):
    co2_prof = np.loadtxt(co2_path, delimiter=",")
    vmr = np.mean(co2_prof)
    co2_prof = co2_vmr / vmr * co2_prof
    n2o_prof = np.loadtxt(n2o_path, delimiter=",")
    vmr = np.mean(n2o_prof)
    n2o_prof = n2o_vmr / vmr * n2o_prof
    ch4_prof = np.loadtxt(ch4_path, delimiter=",")
    vmr = np.mean(ch4_prof)
    ch4_prof = ch4_vmr / vmr * ch4_prof
    
    return co2_prof, n2o_prof, ch4_prof

def estimate_albedo_from_era5_radiation(era5_albedo, time, use_era5_albedo=True):
    if use_era5_albedo:
        idx = np.where((np.array(era5_albedo['time']) >= np.datetime64(time)) & \
                       (np.array(era5_albedo['time']) < np.datetime64(time+dt.timedelta(hours=1))))[0]
        albedo_from_era5_radiation = (np.float(era5_albedo['msdwswrf_NON_CDM'].iloc[idx])-\
                                      np.float(era5_albedo['msnswrf_NON_CDM'].iloc[idx]))/ \
                                      np.float(era5_albedo['msdwswrf_NON_CDM'].iloc[idx])
    else:
        albedo_from_era5_radiation = -1000.0
    return albedo_from_era5_radiation
    
def scale_to_era5_pressure(era5_sfc_pressure, time, use_era5_pressure):
    if use_era5_pressure:
        idx = np.where((np.array(era5_sfc_pressure['time']) >= np.datetime64(time)) & \
                       (np.array(era5_sfc_pressure['time']) < np.datetime64(time+dt.timedelta(hours=1))))[0]
        pressure = np.float(era5_sfc_pressure['pressure'].iloc[idx])
    else:
        pressure = -1013.0
    return pressure

def read_era5_csv(path_albedo, path_sfc_pres):
    era5_albedo = pd.read_csv(path_albedo, parse_dates=[16])
    era5_sfc_pressure = pd.read_csv(path_sfc_pres, parse_dates=[3])
    
    return [era5_albedo, era5_sfc_pressure]

def read_era5_cloud_fraction(path_clt, use_era5_clt):
    if use_era5_clt:
        era5_clt = np.float(np.loadtxt(path_clt))
    else:
        era5_clt = 1.0
    return era5_clt

def scale_era5_to_nearest_cwp(era5_albedo, time, cwp, scale_era5_nearest):
    if scale_era5_nearest:
        idx = np.where((np.array(era5_albedo['time']) >= np.datetime64(time)) & \
                       (np.array(era5_albedo['time']) < np.datetime64(time+dt.timedelta(hours=1))))[0]  

        era5_cwp = np.float(era5_albedo['cwp(gm-2)'].iloc[idx])
        cwp = cwp * (era5_cwp/np.sum(cwp))       
    return cwp

def reff_IFS(lwc, lat, lon, time):
    v10 = interpolate_era5_v2.interpolate_ERA5("/home/phi.richter/Data/ERA5_nc/ERA5_wind_speed.nc", 'v10', [lat], [lon], [time])[-1]
    u10 = interpolate_era5_v2.interpolate_ERA5("/home/phi.richter/Data/ERA5_nc/ERA5_wind_speed.nc", 'u10', [lat], [lon], [time])[-1]
    ws = np.float(np.sqrt(v10**2 + u10**2))
    if ws < 15:
        a = 0.16
        b = 1.45
    else:
        a = 0.13
        b = 1.89
    c = 1.2
    d = 0.5
    q_a = np.exp(a * ws + b)
    if q_a > 327:
        q_a = 327
    N_a = 10**(c + d * np.log(q_a)/np.log(10))
    rho_wat = 1e6
    k = 0.77
    N_d = -1.15e-3 * N_a**2 + 0.963 * N_a + 5.3
    reff = (3*lwc/(4 * np.pi * rho_wat * k * (1e6*N_d)))**(1.0/3.0)
    return reff*1e6

def rice_IFS(iwc, temp): 
    a = 1.2351
    b = 0.0105 * (temp - 273.15)
    c = 45.8966 * iwc ** 0.2214
    d = 0.7957 * iwc ** 0.2535
    e = temp - 83.15
    deff = (a + b + (c + d * e))
    return 3 * np.sqrt(3) / 8 * deff

def recalculate_according_to_IFS_swap(plev, p_surf, recalc_list):
    pres_new = []
    recalc_new = []
    for lay in range(36, -1, -1):
        pres_k_1 = plev[lay]
        if lay == 0:
            pres_k = p_surf
        else:
            pres_k = plev[lay-1]
        recalc_k_1 = recalc_list[lay]
        if lay == 0:
            recalc_f = interp1d(pres_new, recalc_new, fill_value="extrapolate")
            recalc_k = recalc_f(p_surf)
        else:
            recalc_k = recalc_list[lay-1]
            
        pres_k_12 = 0.5*(pres_k + pres_k_1)
        recalc_k_12 = recalc_k * pres_k * (pres_k_1 - pres_k_12) / (pres_k_12 * (pres_k_1 - pres_k)) + \
                    recalc_k_1 * pres_k_1 * (pres_k_12 - pres_k) / (pres_k_12 * (pres_k_1 - pres_k))
        recalc_new.append(recalc_k_12)
        pres_new.append(pres_k_12)
        
    return np.flip(np.array(recalc_new), axis=-1)