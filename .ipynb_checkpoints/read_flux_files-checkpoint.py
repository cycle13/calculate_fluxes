import subprocess
import sys
import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import datetime as dt

path = sys.argv[1]#os.getenv('HOME') + "/GitHub/cloud_radiative_effects_from_measurements/calculate_fluxes/RRTMG_Cloudnet/"
fname_out = sys.argv[2]#os.getenv('HOME') + "/GitHub/cloud_radiative_effects_from_measurements/calculate_fluxes/RRTMG_Cloudnet/results.csv"
files = sorted(os.listdir(path))

out = {'Albedo(1)': [], \
       'SIC(1)': [], \
       'cwp(gm-2)': [], \
       'down_flux_all_lw(Wm-2)': [], \
       'delta_down_flux_all_lw(Wm-2)': [], \
       'down_flux_clear_lw(Wm-2)': [], \
       'down_flux_all_sw(Wm-2)': [], \
       'delta_down_flux_all_sw(Wm-2)': [], \
       'up_flux_all_lw(Wm-2)': [], \
       'delta_up_flux_all_lw(Wm-2)': [], \
       'up_flux_clear_lw(Wm-2)': [], \
       'up_flux_all_sw(Wm-2)': [], \
       'delta_up_flux_all_sw(Wm-2)': [], \
       'down_flux_clear_sw(Wm-2)': [], \
       'up_flux_clear_sw(Wm-2)': [], \
       'direct_flux_all_sw(Wm-2)': [], \
       'direct_flux_clear_sw(Wm-2)': [], \
       'net_cre_lw(Wm-2)': [], \
       'net_cre_sw(Wm-2)': [], \
       'cre(Wm-2)': [], \
       'down_cre_lw(Wm-2)': [], \
       'down_cre_sw(Wm-2)': [], \
       'latitude': [], \
       'longitude': [], \
       'ri(um)': [], \
       'rl(um)': [], \
       't_surf(K)': [], \
       'sza': [], \
       'time': [], \
       'wpi(1)': [], \
       'dCWP(gm-2)': [], \
       'diff_lwp(gm-2)': [], \
       'dWPi(1)': [], \
       'drl(um)': [], \
       'dri(um)': []}

path_diff_lwp = path#"Cloudnet/"
diff_lwp = pd.DataFrame()
for file_ in sorted(os.listdir(path_diff_lwp)):
    if "diff_lwp" in file_:
        diff_lwp = pd.concat((diff_lwp, pd.read_csv(path_diff_lwp + file_, names=["fname", "diff_lwp"])))
            
for file_ in files:
    if not ".nc" in file_:
        continue
    try:
        with nc.Dataset(path + file_) as f:
            lvl = 0
            idx_diff_lwp = np.where(diff_lwp['fname'] == file_)[0]
            diff_lwp_lwc = np.mean(diff_lwp['diff_lwp'].iloc[idx_diff_lwp])
            albedo = np.float(f.variables['sw_broadband_surface_albedo_direct_radiation'][:])
            iceconc = np.float(f.variables['sea_ice_concentration'][:])
            cwp = np.sum(f.variables['CWP'][:])
            cloud = np.where(f.variables['CWP'][:] > 0.0)
            wpi = np.mean(f.variables['WPi'][:])
            rliq = np.mean(f.variableſ['rl'][:])
            rice = np.mean(f.variables['ri'][:])
            lat = np.float(f.variables['latitude'][:])
            lon = np.float(f.variables['longitude'][:])
            sza = np.float(f.variables['solar_zenith_angle'][:])
            year = int(file_.split("_")[1][:4])
            month = int(file_.split("_")[1][4:6])
            day = int(file_.split("_")[1][6:])
            hour = int(file_.split("_")[2][:2])
            minute = int(file_.split("_")[2][2:4])
            second = int(file_.split("_")[2][4:])
            time = dt.datetime(year, month, day, hour, minute, second)
            #if time < dt.datetime(2017, 7, 14, 6) or time > dt.datetime(2017, 7, 14, 7):
            #    continue
            #print(cwp, f.variables['delta_CWP'][:])
            #time = dt.datetime.strptime(file_, "fluxes_TCWret_results_%Y%m%d%H%M%S.nc")
            down_flux_all_lw = f.variables['all_lw_DOWNWARD FLUX'][lvl]
            down_flux_clear_lw = f.variables['clear_lw_DOWNWARD FLUX'][lvl]
            up_flux_all_lw = f.variables['all_lw_UPWARD FLUX'][lvl]
            up_flux_clear_lw = f.variables['clear_lw_UPWARD FLUX'][lvl]
            down_flux_all_sw = f.variables['all_sw_DOWNWARD FLUX'][lvl]
            down_flux_clear_sw = f.variables['clear_sw_DOWNWARD FLUX'][lvl]
            up_flux_all_sw = f.variables['all_sw_UPWARD FLUX'][lvl]
            up_flux_clear_sw = f.variables['clear_sw_UPWARD FLUX'][lvl]
            dir_flux_all_sw = f.variables['all_sw_DIRDOWN FLUX'][lvl]
            dir_flux_clear_sw = f.variables['clear_sw_DIRDOWN FLUX'][lvl]
            t_surf = f.variables['temperature'][0]

            if rliq.size == 0:
                rliq = -1
            if rice.size == 0:
                rice = -1

            delta_cwp_down_flux_all_lw = np.abs(f.variables['difference_quotient_cwp_lw_DOWNWARD FLUX'][lvl])
            delta_cwp_down_flux_all_sw = np.abs(f.variables['difference_quotient_cwp_sw_DOWNWARD FLUX'][lvl])
            delta_rl_down_flux_all_lw = np.abs(f.variables['difference_quotient_rl_lw_DOWNWARD FLUX'][lvl])
            delta_rl_down_flux_all_sw = np.abs(f.variables['difference_quotient_rl_sw_DOWNWARD FLUX'][lvl])
            delta_ri_down_flux_all_lw = np.abs(f.variables['difference_quotient_ri_lw_DOWNWARD FLUX'][lvl])
            delta_ri_down_flux_all_sw = np.abs(f.variables['difference_quotient_ri_sw_DOWNWARD FLUX'][lvl])
            delta_wpi_down_flux_all_lw = np.abs(f.variables['difference_quotient_wpi_lw_DOWNWARD FLUX'][lvl])
            delta_wpi_down_flux_all_sw = np.abs(f.variables['difference_quotient_wpi_sw_DOWNWARD FLUX'][lvl])

            delta_cwp_up_flux_all_lw = np.abs(f.variables['difference_quotient_cwp_lw_UPWARD FLUX'][lvl])
            delta_cwp_up_flux_all_sw = np.abs(f.variables['difference_quotient_cwp_sw_UPWARD FLUX'][lvl])
            delta_rl_up_flux_all_lw = np.abs(f.variables['difference_quotient_rl_lw_UPWARD FLUX'][lvl])
            delta_rl_up_flux_all_sw = np.abs(f.variables['difference_quotient_rl_sw_UPWARD FLUX'][lvl])
            delta_ri_up_flux_all_lw = np.abs(f.variables['difference_quotient_ri_lw_UPWARD FLUX'][lvl])
            delta_ri_up_flux_all_sw = np.abs(f.variables['difference_quotient_ri_sw_UPWARD FLUX'][lvl])
            delta_wpi_up_flux_all_lw = np.abs(f.variables['difference_quotient_wpi_lw_UPWARD FLUX'][lvl])
            delta_wpi_up_flux_all_sw = np.abs(f.variables['difference_quotient_wpi_sw_UPWARD FLUX'][lvl])
            dcwp = np.sum(f.variables['delta_CWP'][:])
            dwpi = np.mean(f.variables['delta_WPi'][:])
            drliq = np.mean(f.variables['delta_rl'][:])
            drice = np.mean(f.variables['delta_ri'][:])

            if np.isinf(dcwp) or np.isinf(dwpi) or np.isinf(drliq) or np.isinf(drice) or \
                np.isnan(dcwp) or np.isnan(dwpi) or np.isnan(drliq) or np.isnan(drice) or \
                np.isnan(delta_cwp_down_flux_all_lw * dcwp + delta_rl_down_flux_all_lw * drliq + delta_ri_down_flux_all_lw * drice + delta_wpi_down_flux_all_lw * dwpi) or np.ma.is_masked(dcwp) or np.ma.is_masked(dwpi) or np.ma.is_masked(drliq) or np.ma.is_masked(drice):
                continue
            out['Albedo(1)'].append(albedo)
            out['SIC(1)'].append(iceconc)
            out['cwp(gm-2)'].append(cwp)
            out['down_flux_all_lw(Wm-2)'].append(down_flux_all_lw)
            out['delta_down_flux_all_lw(Wm-2)'].append(delta_cwp_down_flux_all_lw * dcwp + delta_rl_down_flux_all_lw * drliq + delta_ri_down_flux_all_lw * drice + delta_wpi_down_flux_all_lw * dwpi)
            out['down_flux_clear_lw(Wm-2)'].append(down_flux_clear_lw)
            out['down_flux_all_sw(Wm-2)'].append(down_flux_all_sw)
            out['delta_down_flux_all_sw(Wm-2)'].append(delta_cwp_down_flux_all_sw * dcwp + delta_rl_down_flux_all_sw * drliq + delta_ri_down_flux_all_sw * drice + delta_wpi_down_flux_all_sw * dwpi)
            out['up_flux_all_lw(Wm-2)'].append(up_flux_all_lw)
            out['down_flux_clear_sw(Wm-2)'].append(down_flux_clear_sw)
            out['delta_up_flux_all_lw(Wm-2)'].append(delta_cwp_up_flux_all_lw * dcwp + delta_rl_up_flux_all_lw * drliq + delta_ri_up_flux_all_lw * drice + delta_wpi_up_flux_all_lw * dwpi)
            out['up_flux_clear_lw(Wm-2)'].append(up_flux_clear_lw)
            out['up_flux_all_sw(Wm-2)'].append(up_flux_all_sw)
            out['delta_up_flux_all_sw(Wm-2)'].append(delta_cwp_up_flux_all_sw * dcwp + delta_rl_up_flux_all_sw * drliq + delta_ri_up_flux_all_sw * drice + delta_wpi_up_flux_all_sw * dwpi)
            out['up_flux_clear_sw(Wm-2)'].append(up_flux_clear_sw)
            out['direct_flux_all_sw(Wm-2)'].append(dir_flux_all_sw)
            out['direct_flux_clear_sw(Wm-2)'].append(dir_flux_clear_sw)
            out['net_cre_lw(Wm-2)'].append(down_flux_all_lw - down_flux_clear_lw - (up_flux_all_lw - up_flux_clear_lw))
            out['net_cre_sw(Wm-2)'].append(down_flux_all_sw - down_flux_clear_sw - (up_flux_all_sw - up_flux_clear_sw))
            out['cre(Wm-2)'].append(out['net_cre_lw(Wm-2)'][-1]+out['net_cre_sw(Wm-2)'][-1])
            out['down_cre_lw(Wm-2)'].append(down_flux_all_lw - down_flux_clear_lw)
            out['down_cre_sw(Wm-2)'].append(down_flux_all_sw - down_flux_clear_sw)
            out['latitude'].append(lat)
            out['longitude'].append(lon)
            out['rl(um)'].append(rliq)
            out['ri(um)'].append(rice)
            out['sza'].append(sza)
            out['time'].append(time)
            out['wpi(1)'].append(wpi)
            out['t_surf(K)'].append(t_surf)
            out['dCWP(gm-2)'].append(dcwp)
            out['dWPi(1)'].append(dwpi)
            out['drl(um)'].append(drliq)
            out['dri(um)'].append(drice)
            out['diff_lwp(gm-2)'].append(diff_lwp_lwc)
    except KeyError:
        print(file_)
        #print(f.variables)
        #exit(-1)
        pass
out = pd.DataFrame(out)

out.to_csv(fname_out, index=False)
subprocess.call(['head', '-n', '1', fname_out])