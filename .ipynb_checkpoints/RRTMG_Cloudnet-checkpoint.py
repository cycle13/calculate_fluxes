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
sys.path.append("/mnt/beegfs/user/phi.richter/ERA5_Home/")
import run_RRTMG_inline as RRTM
import datetime as dt
import pandas as pd
import interpolate_era5
import read_era5_atmosphere
import math_manipulations
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
        
def main(month, \
         day, \
         path, \
         out, \
         semiss, \
         scale_era5_nearest, \
         use_era5_clt):
    
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

    ## Read Trace Gases

        
    for file_ in sorted(os.listdir(path)):
        co2_prof, n2o_prof, ch4_prof = math_manipulations.read_trace_gases(os.getenv('HOME') + \
                                                '/Data/input/co2.csv', \
                                                co2_vmr, \
                                                os.getenv('HOME') + \
                                                '/Data/input/n2o.csv', \
                                                n2o_vmr, \
                                                os.getenv('HOME') + \
                                                '/Data/input/ch4.csv', \
                                                ch4_vmr)
        if ".csv" not in file_:
            continue

        time_str = file_.split("_")[1] + file_.split("_")[2]
        time = dt.datetime.strptime(time_str, "%Y%m%d%H%M%S")

        if time < dt.datetime(2017, month, day):
            continue
        elif time > dt.datetime(2017, month, day) + dt.timedelta(days=1):
            break
        

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
  

        atm = pd.read_csv(path + file_)
        
        ## Perform averaging
        temp_av = np.array([np.array(atm['temperature(K)'])[0]])
        height_av = np.array([np.array(atm['height(m)'])[0]*1e-3])
        humd_av = np.array([np.array(atm['humidity(%)'])[0]])
        plev_av = np.array([np.array(atm['pressure(hPa)'])[0]])
        lwc_av = np.array([np.array(atm['lwc(kgm-3)'])[0]])
        iwc_av = np.array([np.array(atm['iwc(kgm-3)'])[0]])
        lwc_err_av = np.array([np.array(atm['lwc_err(dB)'])[0]])
        iwc_err_av = np.array([np.array(atm['iwc_err(dB)'])[0]])
        rliq_av = np.array([np.array(atm['rliq(um)'])[0]])
        rice_av = np.array([np.array(atm['rice(um)'])[0]])
        rliq_err_av = np.array([np.array(atm['rliq_err(um)'])[0]])
        rice_err_av = np.array([np.array(atm['rice_err(um)'])[0]])
        diff_lwp = np.float(atm['diff_lwc_lwp(gm-2)'].iloc[0])
        step = 7
        for ii in range(1, np.array(atm['temperature(K)']).size-step-1, step):
            temp_av = np.concatenate((temp_av, [np.mean(np.array(atm['temperature(K)'])[ii:ii+step])]))
            height_av = np.concatenate((height_av, [np.mean(np.array(atm['height(m)'])[ii:ii+step]*1e-3)]))
            humd_av = np.concatenate((humd_av, [np.mean(np.array(atm['humidity(%)'])[ii:ii+step])]))
            plev_av = np.concatenate((plev_av, [np.mean(np.array(atm['pressure(hPa)'])[ii:ii+step])]))       
            lwc_av = np.concatenate((lwc_av, [1e3*np.mean(np.array(atm['lwc(kgm-3)'])[ii:ii+step])]))
            iwc_av = np.concatenate((iwc_av, [1e3*np.mean(np.array(atm['iwc(kgm-3)'])[ii:ii+step])]))
            lwc_err_av = np.concatenate((lwc_err_av, [np.mean(np.array(atm['lwc_err(dB)'])[ii:ii+step])]))
            iwc_err_av = np.concatenate((iwc_err_av, [np.mean(np.array(atm['iwc_err(dB)'])[ii:ii+step])]))
            
            idx = np.where(np.array(atm['lwc(kgm-3)'])[ii:ii+step] > 0.0)[0]
            idx = idx+ii
            rliq_av = np.concatenate((rliq_av, [np.mean(np.array(atm['rliq(um)'])[idx])]))
            rliq_err_av = np.concatenate((rliq_err_av, [np.mean(np.array(atm['rliq_err(um)'])[idx])]))
            idx = np.where(np.array(atm['iwc(kgm-3)'])[ii:ii+step] > 0.0)[0]
            idx = idx+ii
            rice_av = np.concatenate((rice_av, [np.mean(np.array(atm['rice(um)'])[idx])]))
            rice_err_av = np.concatenate((rice_err_av, [np.mean(np.array(atm['rice_err(um)'])[idx])]))

        for ii in range(np.array(atm['temperature(K)']).size-step, np.array(atm['temperature(K)']).size, 1):
            temp_av = np.concatenate((temp_av, [np.mean(np.array(atm['temperature(K)'])[ii])]))
            humd_av = np.concatenate((humd_av, [np.mean(np.array(atm['humidity(%)'])[ii])]))
            plev_av = np.concatenate((plev_av, [np.mean(np.array(atm['pressure(hPa)'])[ii])]))
            height_av = np.concatenate((height_av, [np.mean(np.array(atm['height(m)'])[ii]*1e-3)]))
            lwc_av = np.concatenate((lwc_av, [1e3*np.mean(np.array(atm['lwc(kgm-3)'])[ii])]))
            iwc_av = np.concatenate((iwc_av, [1e3*np.mean(np.array(atm['iwc(kgm-3)'])[ii])]))
            lwc_err_av = np.concatenate((lwc_err_av, [np.mean(np.array(atm['lwc_err(dB)'])[ii])]))
            iwc_err_av = np.concatenate((iwc_err_av, [np.mean(np.array(atm['iwc_err(dB)'])[ii])]))
            rliq_av = np.concatenate((rliq_av, [np.mean(np.array(atm['rliq(um)'])[ii])]))
            rliq_err_av = np.concatenate((rliq_err_av, [np.mean(np.array(atm['rliq_err(um)'])[ii])]))
            rice_av = np.concatenate((rice_av, [np.mean(np.array(atm['rice(um)'])[ii])]))
            rice_err_av = np.concatenate((rice_err_av, [np.mean(np.array(atm['rice_err(um)'])[ii])]))
 
        idx_cloud = np.where((lwc_av > 0.0) | (iwc_av > 0.0))[0]
        dz_av = np.concatenate((np.diff(height_av), [0]))

        ## Calculate absolute error
        iwc_err_abs = iwc_av[idx_cloud]*10**iwc_err_av[idx_cloud]*1e-2#1e-2 wegen Prozent
        lwc_err_abs = lwc_av[idx_cloud]*10**lwc_err_av[idx_cloud]*1e-2

        lwp = lwc_av[idx_cloud] * dz_av[idx_cloud]*1e3
        iwp = iwc_av[idx_cloud] * dz_av[idx_cloud]*1e3
        
        ## Error of IWP and LWP
        diwp = iwc_err_abs * dz_av[idx_cloud]*1e3
        dlwp = lwc_err_abs * dz_av[idx_cloud]*1e3

        
        height = height_av
        temp = temp_av
        humd = humd_av
        plev = plev_av
        rliq = rliq_av[idx_cloud]
        rice = rice_av[idx_cloud]                
        drliq = rliq_err_av[idx_cloud]
        for ii in range(len(drliq)):
            if np.isnan(drliq[ii]):
                drliq[ii] = 0.0
        drice = rice_err_av[idx_cloud]
        for ii in range(len(drice)):
            if np.isnan(drice[ii]):
                drice[ii] = 0.0
        cwp = lwp + iwp
        wpi = iwp / cwp
        
        idx_noice = np.where(wpi == 0.0)[0]
        if idx_noice.size != 0:
            rice[idx_noice] = 40
        idx_noliq = np.where(wpi == 1.0)[0]
        if idx_noliq.size != 0:
            rliq[idx_noliq] = 10
        ## Error of CWP
        dcwp = np.abs(dlwp) + np.abs(diwp)
        
        ## Error of WPi
        dwpi = np.abs(diwp / cwp) + np.abs(dcwp * iwp / cwp**2)

        for ii in range(len(dcwp)):
            if np.isnan(dcwp[ii]):
                dcwp[ii] = 0.0
        for ii in range(len(dwpi)):
            if np.isnan(dwpi[ii]):
                dwpi[ii] = 0.0
 
        prof_height = np.loadtxt(os.getenv('HOME') + "/Data/input/z.csv", delimiter=",")
        if co2_prof.size != height.size:
            co2_prof = np.interp(height, prof_height, co2_prof)
            n2o_prof = np.interp(height, prof_height, n2o_prof)
            ch4_prof = np.interp(height, prof_height, ch4_prof)

                
        ## Use the albedo estimated from the ERA5 radiation   
        albedo = math_manipulations.estimate_albedo_from_era5_radiation(era5_albedo, time)
        
        try:
            clt = math_manipulations.read_era5_cloud_fraction(path_clt + \
                                   "era5_clt_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}_{}_{}.csv".format(year,month,day,hour,minute,sec,lat,lon), \
                                   use_era5_clt)
        except Exception:
            continue
            
        cwp = math_manipulations.scale_era5_to_nearest_cwp(era5_albedo, time, cwp, scale_era5_nearest)
        
        atm_era5 = read_era5_atmosphere.read_era5_atmosphere(lat, lon, dt.datetime(year, month, day, hour))#+dt.timedelta(hours=1))
        #print(height)
        #plt.plot(height, humd, "x-")
        #plt.plot(atm_era5['height'], atm_era5['rh'], "x-")
        #print(atm_era5['height'])
        temp_f = interp1d(atm_era5['height'], atm_era5['temperature'], fill_value="extrapolate")
        pres_f = interp1d(atm_era5['height'], atm_era5['pressure'], fill_value="extrapolate")
        humd_f = interp1d(atm_era5['height'], atm_era5['rh'], bounds_error=False, fill_value=humd[0])
        temp = temp_f(height)
        plev = pres_f(height)*10
        humd = humd_f(height)
        #plt.plot(height, humd, "x-")
        #plt.grid(True)
        #plt.savefig("grid.png", dpi=300)
        #exit(-1)
        oob = np.where((np.array(rliq) < 2.5) & \
                                (wpi < 1.0))[0].size
        oob+= np.where((np.array(rliq) >60.0) & \
                                (wpi < 1.0))[0].size
        oob+= np.where((np.array(rice) < 13.0) & \
                                (wpi > 0.0))[0].size
        oob+= np.where((np.array(rice) > 131.0) & \
                                (wpi > 0.0))[0].size

        rice[np.where(wpi == 0.0)[0]] = 40.0
        rliq[np.where(wpi == 1.0)[0]] = 5.0
        if oob != 0:
            continue

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
    semiss = float(sys.argv[5])#0.98
    
    ## Scale to ERA5 using the nearest value of ERA5 to the chosen 
    ## temporal and spatial coordinate. Use entire atmospheric column
    scale_era5_nearest = bool(int(sys.argv[6]))#False
    
    ## Use cloud fraction given by ERA5
    use_era5_clt = bool(int(sys.argv[7]))#True
    
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    os.chdir(path_out)

    main(month, day, path_atm, path_out, \
         semiss, scale_era5_nearest, use_era5_clt)
