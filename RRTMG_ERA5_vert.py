# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:20:46 2020

@author: Philipp
"""
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import numpy as np
#sys.path.append(os.getenv('HOME') + "/GitHub/cloud_radiative_effects_from_measurements/RRTMG/")
sys.path.append("/mnt/beegfs/user/phi.richter/ERA5_Home/")
import run_RRTMG_inline as RRTM
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d
import interpolate_era5
import math_manipulations

CLIM_RLIQ = 8.0
CLIM_RICE = 20.0



def convert(humd, temp, pres):
    
    R_S = 461.5#J/kg*K
    temp = temp + 273.15
    pres = pres * 100
    humd = humd / 1000.0
    mixing_ratio = ((1.0/humd) - 1)**(-1)
    vapour_pressure = mixing_ratio * pres / (mixing_ratio + 0.622)
    ah = vapour_pressure / (R_S * temp)
    
    return ah

def read_era5_csv(path_albedo, path_sfc_pres):
    era5_albedo = pd.read_csv(path_albedo, parse_dates=[16])
    era5_sfc_pressure = pd.read_csv(path_sfc_pres, parse_dates=[3])
    
    return [era5_albedo, era5_sfc_pressure]

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

def read_era5_cwc(path_era5):
    era5_pres = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0, 800.0, 775.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 225.0, 200.0, 175.0, 150.0, 125.0, 100.0, 70.0, 50.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0])  
    era5_cwc = pd.read_csv(path_era5)
    
    era5_lwc = np.array(era5_cwc['lwc(kg/kg)'])*1e3
    era5_iwc = np.array(era5_cwc['iwc(kg/kg)'])*1e3

    era5_temp= np.array(era5_cwc['temperature(K)'])
    era5_height = np.array(era5_cwc['height(m)'])
    era5_humd = np.array(era5_cwc['humidity(%)'])
    era5_lwc = convert(era5_lwc, era5_temp, era5_pres)*1e3
    era5_iwc = convert(era5_iwc, era5_temp, era5_pres)*1e3
    
    dz = np.concatenate(([era5_height[0]], np.diff(era5_height)))
    era5_lwp = era5_lwc*dz
    era5_iwp = era5_iwc*dz
    era5_cwp = era5_lwp+era5_iwp
    era5_wpi = np.zeros(era5_cwp.size)
    
    idx_cld = np.where(era5_cwp > 0.0)[0]
    if idx_cld.size > 0:
        era5_wpi[idx_cld] = era5_iwp[idx_cld]/era5_cwp[idx_cld]
    return era5_cwp, era5_wpi, era5_height, era5_humd, era5_temp, era5_pres, era5_lwc, era5_iwc

def only_lowest_level(cwp, wpi, rliq, rice, idx_cloud, height_offset, lowest_level):
    if lowest_level:
        cwp = np.array([cwp[0]])
        wpi = np.array([wpi[0]])
        rliq = np.array([rliq[0]])
        rice = np.array([rice[0]])
        idx_cloud = np.array([idx_cloud[0]+height_offset])
    return cwp, wpi, rliq, rice, idx_cloud

def scale_era5_to_nearest_cwp(era5_albedo, time, cwp, scale_era5_nearest):
    if scale_era5_nearest:
        idx = np.where((np.array(era5_albedo['time']) >= np.datetime64(time)) & \
                       (np.array(era5_albedo['time']) < np.datetime64(time+dt.timedelta(hours=1))))[0]  

        era5_cwp = np.float(era5_albedo['cwp(gm-2)'].iloc[idx])
        #if iwc+lwc > 0.0:
        #    era5_wpi = iwc/(lwc+iwc) * np.ones(cwp.size)
        #else:
        #    era5_wpi = np.zeros(cwp.size)
        cwp = cwp * (era5_cwp/np.sum(cwp))       
    return cwp

def estimate_albedo_from_era5_radiation(era5_albedo, time, use_era5_albedo):
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

def read_era5_cloud_fraction(path_clt, use_era5_clt):
    if use_era5_clt:
        era5_clt = np.float(np.loadtxt(path_clt))
    else:
        era5_clt = 1.0
    return era5_clt
    
        
def main(month, \
         day, \
         path, \
         out, \
         semiss, \
         scale_era5_nearest):
    
    co2_vmr = 410.
    n2o_vmr = 0.332
    ch4_vmr = 1.860


    ## Create working directory
    wdir = "{}_{}/".format(month, day)
    if os.path.exists(wdir):
        shutil.rmtree(wdir)
    os.mkdir(wdir)
    os.chdir(wdir)
    
    ## Read ERA5 data
    path_albedo  = os.getenv('HOME') + \
                  "/Data/ERA5_csv/era5_nearest_RAW.csv"
    path_sfc_pres = os.getenv('HOME') + "/Data/ERA5_csv/pressure_era5_nearest_RAW.csv.csv"
    path_clt = os.getenv('HOME') + "/Data/ERA5_csv/clt/"
    era5_csv = math_manipulations.read_era5_csv(path_albedo, path_sfc_pres)
    era5_albedo = era5_csv[0]
    era5_sfc_pressure = era5_csv[1]

    ## Read Trace Gases
    co2_prof, n2o_prof, ch4_prof = math_manipulations.read_trace_gases(os.getenv('HOME') + \
                                                    '/Data/input/co2.csv', \
                                                    co2_vmr, \
                                                    os.getenv('HOME') + \
                                                    '/Data/input/n2o.csv', \
                                                    n2o_vmr, \
                                                    os.getenv('HOME') + \
                                                    '/Data/input/ch4.csv', \
                                                    ch4_vmr)
        
    for file_ in sorted(os.listdir(path)):
        if ".csv" not in file_:
            continue

        time_str = file_.split("_")[1] + file_.split("_")[2]
        time = dt.datetime.strptime(time_str, "%Y%m%d%H%M%S")
        if time < dt.datetime(2017, month, day):
            continue
        elif time > dt.datetime(2017, month, day) + dt.timedelta(days=1):
            break
        
        ## Read Cloud Water Paths from ERA5

        parsed = math_manipulations.parse_date_and_sza(file_)
        lat = parsed[0]
        lon = parsed[1]
        sza = parsed[2]
        year = parsed[3]
        month = parsed[4]
        day = parsed[5]
        hour = parsed[6]
        minute = parsed[7]
        sec = parsed[8]
                     
        path_era5 = "/home/phi.richter/Data/ERA5_csv/CWC/ERA5_CWC_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}_{}_{}.csv".format(year,month,day,hour,minute,sec,lat,lon)

        ## Use Cloud Water Paths and atmosphere from ERA5
        if os.path.exists(path_era5):
            cwp, wpi, height, humd, temp, plev, era5_lwc, era5_iwc = read_era5_cwc(path_era5)
        else:
            continue
        dcwp = np.zeros(cwp.size)
        dwpi = np.zeros(cwp.size)
        drliq = np.zeros(cwp.size)
        drice = np.zeros(cwp.size)
        rliq = math_manipulations.reff_IFS(era5_lwc, lat, lon, np.datetime64("{:04d}-{:02d}-{:02d}T{:02d}:00:00".format(year, month, day, hour+1)))
        rice = math_manipulations.rice_IFS(era5_iwc, temp)
        height *= 1e-3
        height = np.around(height, decimals=2)
        idx_cloud = np.where(cwp > 0.0)[0]
        ## Recalculate temperature according to IFS documentation
        p_surf = scale_to_era5_pressure(era5_sfc_pressure, time, True)
        humd = math_manipulations.recalculate_according_to_IFS_swap(plev, p_surf, humd)
        temp = math_manipulations.recalculate_according_to_IFS_swap(plev, p_surf, temp)

        prof_height = np.loadtxt(os.getenv('HOME') + "/Data/input/z.csv", delimiter=",")
        if co2_prof.size != cwp.size:
            co2_prof = np.interp(height, prof_height, co2_prof)
            n2o_prof = np.interp(height, prof_height, n2o_prof)
            ch4_prof = np.interp(height, prof_height, ch4_prof)
        diff_lwp = 0
        cwp = cwp[idx_cloud]
        wpi = wpi[idx_cloud]
        rliq = rliq[idx_cloud]
        rice = rice[idx_cloud]
        idx_rliq = np.where(rliq > 30.0)[0]
        if idx_rliq.size != 0:
            rliq[idx_rliq] = 30.0
        idx_rliq = np.where(rliq < 4.0)[0]
        if idx_rliq.size != 0:
            rliq[idx_rliq] = 4.0
        idx_rice = np.where(rice > 155.0)[0]
        if idx_rice.size != 0:
            rice[idx_rice] = 155.0
        idx_rice = np.where(rice < 20+40*np.cos(lat*np.pi/180))[0]
        if idx_rice.size != 0:
            rice[idx_rice] = 20+40*np.cos(lat*np.pi/180)
        dcwp = np.zeros(cwp.size)
        dwpi = np.zeros(cwp.size)
        drliq = np.zeros(cwp.size)
        drice = np.zeros(cwp.size)

            
        ## Scale the water path to the values of ERA5
        cwp = math_manipulations.scale_era5_to_nearest_cwp(era5_albedo, time, cwp, scale_era5_nearest)
        
                
        ## Use the albedo estimated from the ERA5 radiation   
        albedo = math_manipulations.estimate_albedo_from_era5_radiation(era5_albedo, time, True)
        
      
        ## Use cloud fraction from ERA5
        try:
            clt = math_manipulations.read_era5_cloud_fraction(path_clt + \
                                           "era5_clt_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}_{}_{}.csv".format(year,month,day,hour,minute,sec,lat,lon), \
                                           True)
        except Exception:
            continue

        rice[np.where(wpi == 0.0)[0]] = 40.0
        rliq[np.where(wpi == 1.0)[0]] = 5.0


        #pressure = -10
        try:

            iceconc = interpolate_era5.interpolate_ERA5("/mnt/beegfs/user/phi.richter/ERA5_data/ERA5_surface_radiation_REANALYSIS.nc", "siconca", lat, lon, time)[0]
            with open("{}/diff_lwp_{}_{}.csv".format(out, month, day), "a") as f:
                f.write("{}.nc,{}\n".format(file_.split(".csv")[0], diff_lwp))
            RRTM.main(lat=lat, \
                        lon=lon, \
                        sza=sza, \
                        cwp=cwp, \
                        wpi=wpi, \
                        rl=rliq, \
                        ri=rice, \
                        cloud=idx_cloud, \
                        z=height, \
                        t=temp, \
                        q=humd, \
                        p=plev, \
                        dcwp=dcwp, \
                        dwpi=dwpi, \
                        drl=drliq, \
                        dri=drice, \
                        co2=co2_prof, \
                        n2o=n2o_prof, \
                        ch4=ch4_prof, \
                        albedo_dir=albedo, \
                        albedo_diff=albedo, \
                        iceconc=iceconc, \
                        semiss=semiss, \
                        clt=clt, \
                        fname=out + "/{}.nc".format(file_.split(".csv")[0]))
        except Exception:
            print("Exception")
            continue

if __name__ == '__main__':
    
    month = int(sys.argv[1])
    day = int(sys.argv[2])
    path_atm = sys.argv[3]
    path_out = sys.argv[4]
    
    ## Broadband surface emissivity for longwave radiation
    semiss = float(sys.argv[5])#0.99
    
    ## Scale to ERA5 using the nearest value of ERA5 to the chosen 
    ## temporal and spatial coordinate. Use entire atmospheric column
    scale_era5_nearest = bool(int(sys.argv[6]))#False

    
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    os.chdir(path_out)

    main(month, day, path_atm, path_out, \
         semiss=semiss, \
         scale_era5_nearest=scale_era5_nearest)
