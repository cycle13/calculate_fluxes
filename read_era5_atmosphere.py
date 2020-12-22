#!/usr/bin/python3

import netCDF4 as nc
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

g0 = 9.80665#ms-2

def read_era5_atmosphere(lat, lon, time):
    time_interpolate = int((time - dt.datetime(1970, 1, 1)).total_seconds())
    with nc.Dataset("/home/phi.richter/Data/ERA5_nc/ERA5_arctic_atmosphere.nc", "r") as f:
        time = f.variables['time'][:]#1-D
        plev = f.variables['plev'][:]#1-D, 37 Level
        latitude = f.variables['lat'][:]#1-D
        longitude = f.variables['lon'][:]#1-D
        rh = f.variables['hurs'][:]#5-D, realization, time, plev, lat, lon
        temperature = f.variables['ta'][:]#5-D, realization, time, plev, lat, lon
        geopotential = f.variables['geopotential'][:]#5-D, realization, time, plev, lat, lon
        
    height_prof_int = np.zeros(len(plev))
    temp_prof_int = np.zeros(len(plev))
    rh_prof_int = np.zeros(len(plev))
    height_prof_near = np.zeros(len(plev))
    temp_prof_near = np.zeros(len(plev))
    rh_prof_near = np.zeros(len(plev))
    
    for level in range(len(plev)):
        value = geopotential[0,:,level,:,:]/g0
        value = value.reshape(time.size, latitude.size, longitude.size)
        f = RegularGridInterpolator((time, latitude, longitude), value)
        height_prof_int[level] = f((time_interpolate, lat, lon), method="linear")
        height_prof_near[level] = f((time_interpolate, lat, lon), method="nearest")
        
        value = temperature[0,:,level,:,:]
        value = value.reshape(time.size, latitude.size, longitude.size)
        f = RegularGridInterpolator((time, latitude, longitude), value)
        temp_prof_int[level] = f((time_interpolate, lat, lon), method="linear")
        temp_prof_near[level] = f((time_interpolate, lat, lon), method="nearest")
        
        value = rh[0,:,level,:,:]
        value = value.reshape(time.size, latitude.size, longitude.size)
        f = RegularGridInterpolator((time, latitude, longitude), value)
        rh_prof_int[level] = f((time_interpolate, lat, lon), method="linear")
        rh_prof_near[level] = f((time_interpolate, lat, lon), method="nearest")
        
    
    return {'time': time_interpolate, 'height': height_prof_int*1e-3, 'rh': rh_prof_int, 'temperature': temp_prof_int, 'pressure': plev*1e-3}
        
if __name__ == '__main__':
    read_era5_atmosphere(80, 0, dt.datetime(2017, 6, 11, 0))
