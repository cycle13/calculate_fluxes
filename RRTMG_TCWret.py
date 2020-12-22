import os
import sys
import shutil
import numpy as np
import netCDF4 as nc
#sys.path.append(os.getenv('HOME') + "/GitHub/cloud_radiative_effects_from_measurements/RRTMG/")
sys.path.append(os.getenv('HOME') + "/GitHub/RESULTS_no_version_control")
sys.path.append("/mnt/beegfs/user/phi.richter/ERA5_Home/")
import interpolate_era5
import read_database as mie
import run_RRTMG_inline as RRTM
import datetime as dt
import pandas as pd

def read_era5_csv(path_albedo):
    era5_albedo = pd.read_csv(path_albedo, parse_dates=[16])
    
    return era5_albedo


def estimate_albedo_from_era5_radiation(era5_albedo, time):
    idx = np.where((np.array(era5_albedo['time']) >= np.datetime64(time)) & \
                   (np.array(era5_albedo['time']) < np.datetime64(time+dt.timedelta(hours=1))))[0]
    albedo_from_era5_radiation = (np.float(era5_albedo['msdwswrf_NON_CDM'].iloc[idx])-\
                                  np.float(era5_albedo['msnswrf_NON_CDM'].iloc[idx]))/ \
                                  np.float(era5_albedo['msdwswrf_NON_CDM'].iloc[idx])
    return albedo_from_era5_radiation

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

path_albedo  = os.getenv('HOME') + \
              "/GitHub/cloud_radiative_effects_from_measurements/DATA/raw/ERA5/era5_nearest_RAW.csv"
era5_csv = read_era5_csv(path_albedo)
era5_albedo = era5_csv

path_results = os.getenv("HOME") + "/GitHub/RESULTS_no_version_control/RESULTS_FIR_11_09_2020/"
files = sorted(os.listdir(path_results))
ssp_liq = os.getenv("HOME") + "/GitHub/RESULTS_no_version_control/ssp_db.mie_wat.gamma_sigma_0p100"
ssp_liq_263 = os.getenv("HOME") + "/GitHub/RESULTS_no_version_control/ssp_db.mie_wat_zasetsky263.gamma_sigma_0p100"
ssp_liq_253 = os.getenv("HOME") + "/GitHub/RESULTS_no_version_control/ssp_db.mie_wat_zasetsky253.gamma_sigma_0p100"
ssp_liq_240 = os.getenv("HOME") + "/GitHub/RESULTS_no_version_control/ssp_db.mie_wat_zasetsky240.gamma_sigma_0p100"
ssp_ice = os.getenv("HOME") + "/GitHub/RESULTS_no_version_control/ssp_db.mie_ice.gamma_sigma_0p100"

co2_vmr = 410.
n2o_vmr = 0.332
ch4_vmr = 1.860
    
co2_prof, n2o_prof, ch4_prof = read_trace_gases(os.getenv('HOME') + \
                                                '/GitHub/run_LBLDIS/input/co2.csv', \
                                                co2_vmr, \
                                                os.getenv('HOME') + \
                                                '/GitHub/run_LBLDIS/input/n2o.csv', \
                                                n2o_vmr, \
                                                os.getenv('HOME') + \
                                                '/GitHub/run_LBLDIS/input/ch4.csv', \
                                                ch4_vmr)

for file_ in files:
    with nc.Dataset(path_results + file_) as f:
        date = dt.datetime.strptime(file_, "results_%Y%m%d%H%M%S.nc")
        temp = f.variables['T'][:]
        plev = f.variables['P'][:]
        humd = f.variables['humidity'][:]
        height = f.variables['z'][:]
        clevel = f.variables['clevel'][:]
        idx_cloud = np.where(height == clevel)[0]
        ctemp = temp[idx_cloud]
        tliq = np.float(f.variables['x_ret'][0])
        tice = np.float(f.variables['x_ret'][1])
        rliq = np.float(f.variables['x_ret'][2])
        rice = np.float(f.variables['x_ret'][3])
        dtliq = np.float(f.variables['x_ret_err'][0])
        dtice = np.float(f.variables['x_ret_err'][1])
        drliq = np.float(f.variables['x_ret_err'][2])
        drice = np.float(f.variables['x_ret_err'][3])
        lat = np.float(f.variables['lat'][:])
        lon = np.float(f.variables['lon'][:])
        sza = np.float(f.variables['sza'][:])
        red_chi_2 = np.float(f.variables['red_chi_2'][:])
        if ctemp > 268.15:
            [liq_db, ice_db] = mie.read_databases(ssp_liq, ssp_ice)
        elif ctemp > 258.15:
            [liq_db, ice_db] = mie.read_databases(ssp_liq_263, ssp_ice)
        elif ctemp > 248.15:
            [liq_db, ice_db] = mie.read_databases(ssp_liq_253, ssp_ice)
        else:
            [liq_db, ice_db] = mie.read_databases(ssp_liq_240, ssp_ice)
        lwp = mie.calc_lwp(rliq, drliq, tliq, dtliq)
        iwp = mie.calc_iwp(tice, dtice, rice, drice, ice_db)
        cwp = lwp[0] + iwp[0]
        dcwp = lwp[1] + iwp[1]
        wpi = iwp[0] / cwp
        dwpi = np.abs(iwp[1]/cwp) + np.abs(iwp[0]*dcwp/cwp**2)
        iceconc = interpolate_era5.interpolate_ERA5("/mnt/beegfs/user/phi.richter/ERA5_data/ERA5_surface_radiation_REANALYSIS.nc", "siconca", lat, lon, date)[0]
        albedo = estimate_albedo_from_era5_radiation(era5_albedo, date)
        
        cwp = np.array([cwp])
        dcwp = np.array([dcwp])
        wpi = np.array([wpi])
        dwpi = np.array([dwpi])
        rliq = np.array([rliq])
        drliq = np.array([drliq])
        rice = np.array([rice])
        drice = np.array([drice])
        
        if red_chi_2 < 0.5 or red_chi_2 > 1.5 or (np.float(rliq) < 2.5 and np.float(wpi) < 1.0) \
            or (np.float(rliq) > 60. and np.float(wpi) < 1.0) \
            or (np.float(rice) < 13. and np.float(wpi) > 0.0) \
            or (np.float(rice) > 131. and np.float(wpi) > 0.0):
            continue
        try:
            flux = RRTM.main(lat=lat, \
                             lon=lon, \
                             sza=sza, \
                             cwp=cwp, \
                             wpi=wpi, \
                             rl=rliq, \
                             ri=rice, \
                             cloud=idx_cloud, \
                             z=height*1e-3, \
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
                             semiss=0.99, \
                             clt=1.0, \
                             fname=os.getenv("HOME") + "/GitHub/cloud_radiative_effects_from_measurements/fluxes/TCWret_FIR/fluxes_TCWret_{}".format(file_))
        except Exception:
            pass