#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:09:11 2020

@author: philipp
"""

#import sys
import os
#sys.path.append("/mnt/beegfs/user/phi.richter/ERA5_Home/")
import subprocess
import pandas as pd
from scipy.io import netcdf_file
import datetime as dt
import numpy as np

path_home = os.getenv("HOME") + "/Code/RRTMG"
SRC_RRTMG_LW = path_home + '/models/rrtmg_lw_v5.00_linux_ifort'
SRC_RRTMG_SW = path_home + '/models/rrtmg_sw_v5.00_linux_pgi'
KEYS_SW = ['LEVEL', 'PRESSURE', 'UPWARD FLUX', 'DIFDOWN FLUX', 'DIRDOWN FLUX', 'DOWNWARD FLUX', 'NET FLUX', 'HEATING RATE']
KEYS_LW = ['LEVEL', 'PRESSURE', 'UPWARD FLUX', 'DOWNWARD FLUX', 'NET FLUX', 'HEATING RATE']

def create_cloud_rrtmg(lay_liq, lay_ice, cwp, rliq, rice, fice, homogenous, clt):
      
    # =============================================================================
    #  RECORD C1.1
    # 	
    #       INFLAG, ICEFLAG, LIQFLAG
    # 
    #            5       10       15
    # 
    #       4X, I1,  4X, I1,  4X, I1
    # 
    # 
    #       Note:  ICEFLAG and LIQFLAG are required only if INFLAG = 2.
    # 
    #             INFLAG = 0 direct specification of optical depths of clouds;
    #                        cloud fraction and cloud optical depth (gray), single scattering albedo,
    # 		       and N-str moments of the phase function
    # 
    #                    = 2 calculation of separate ice and liquid cloud optical depths, with
    #                        parameterizations determined by values of ICEFLAG and LIQFLAG. 
    #                        Cloud fraction, cloud water path, cloud ice fraction, and
    #                        effective ice radius are input for each cloudy layer for all 
    #                        parameterizations.  If LIQFLAG = 1, effective liquid droplet radius
    #                        is also needed.  If ICEFLAG = 1, generalized effective size is
    # 		       is also needed.
    INFLAG = 2
    # 
    #             ICEFLAG = 0 inactive
    #                     = 1 the optical depths (non-gray) due to ice clouds are computed as closely as
    #                         possible to the method in E.E. Ebert and J.A. Curry, JGR, 97, 3831-3836 (1992).
    #                     = 2 the optical properties are computed by a method based on the parameterization
    #                         of spherical ice particles in the RT code, STREAMER v3.0 (Reference: 
    #                         Key. J., Streamer User's Guide, Cooperative Institute for
    #                         Meteorological Satellite Studies, 2001, 96 pp.).
    #                     = 3 the optical depths are computed by a method based on the parameterization
    # 			of ice clouds due to Q. Fu, J. Clim., 9, 2058 (1996).
    ICEFLAG = 1
    # 
    #             LIQFLAG = 0 inactive
    #                     = 1 the optical depths (non-gray) due to water clouds are computed by a method
    #                         based on the parameterization of water clouds due to Y.X. Hu and K. Stamnes,
    #                         J. Clim., 6, 728-742 (1993).
    LIQFLAG = 1
    #                        
    # 	     These methods are further detailed in the comments in the file 'rrtmg_sw_cldprop.F90'
    #              and the module 'rrtmg_sw_susrtop.F90'.
    # 
    RECORD_C1_1  = 4 * " " + "{:1d}".format(INFLAG)
    RECORD_C1_1 += 4 * " " + "{:1d}".format(ICEFLAG)
    RECORD_C1_1 += 4 * " " + "{:1d}".format(LIQFLAG)

    #  RECORD C1.3  (one record for each cloudy layer, INFLAG = 2)
    # 
    #       TESTCHAR,    LAY, CLDFRAC,   TAUCLD or CWP, FRACICE, EFFSIZEICE, EFFSIZELIQ
    # 
    #              1,    3-5,    6-15,           16-25,   26-35,     36-45,     46-55
    # 
    #             A1, 1X, I3,   E10.5,           E10.5,   E10.5,     E10.5,     E10.5
    # 
    # 
    #             TESTCHAR   control character -- if equal to '%', cloud input processing
    #                        is terminated
    # 
    #             LAY        layer number of cloudy layer.  The layer numbering refers to the
    #                        ordering for the upward radiative transfer, i.e. botton to top.
    #                        For IATM = 0 (Record 1.2), each layer's number is equal to the  
    #                        position of its Record 2.1.1 in the grouping of these records.
    #                        For example, the second Record 2.1.1 occurring after Record 2.1
    #                        corresponds to the second layer.  For IATM = 1 (Record 1.2) and 
    #                        IBMAX > 0 (Record 3.1), layer n corresponds to the region between 
    #                        altitudes n and n+1 in the list of layer boundaries in Record 3.3B.  
    #                        For IATM = 1 (Record 1.2) and IBMAX = 0 (Record 3.1), the layer 
    #                        numbers can be determined by running RRTM for the cloudless case
    #                        and examining the TAPE6 output from this run.
    #                        
    #             CLDFRAC    cloud fraction for the layer.
    # 
    #             TAUCLD     (INFLAG = 0 only) total (ice and water) optical depth for the layer
    #      or     CWP        (INFLAG > 0) cloud water path for the layer (g/m2)
    # 
    #             FRACICE    (INFLAG = 2) fraction of the layer's cloud water path in the form
    #                        of ice particles  
    # 
    #             EFFSIZEICE (INFLAG = 2 and ICEFLAG = 1) Effective radius of spherical  
    # 	               ice crystals with equivalent projected area to hexagonal ice particles
    # 	               following Ebert and Curry (1992).
    # 	               Valid sizes are 13.0 - 130.0 microns. 
    #                   
    #                        (INFLAG = 2 and ICEFLAG = 2) Effective radius of spherical  
    # 	               ice crystals, re (see STREAMER manual for definition of this parameter)
    # 	               Valid sizes are 5.0 - 131.0 microns. 
    #                   
    #                        (INFLAG = 2 and ICEFLAG = 3) Generalized effective size of hexagonal
    # 	               ice crystals, dge (see Q. Fu, 1996, for definition of this parameter)
    # 	               Valid sizes are 5.0 - 140.0 microns.  
    # 
    # 		       NOTE: The size descriptions for effective radius and generalized effective
    # 		       size are NOT equivalent.  See the particular references for the appropriate 
    # 		       definition.
    # 
    #             EFFSIZELIQ (INFLAG = 2 and LIQFLAG = 1) Liquid droplet effective radius, re (microns) 
    #                        Valid sizes are 2.5 - 60.0 microns.
    CLDFRAC = clt
    FRACICE = fice
    EFFSIZEICE = rice
    EFFSIZELIQ = rliq
    RECORD_C1_3 = ""
    CWP = cwp
    ii = 0
    f = open("log", "a")
    f.write("{}\n".format(CWP))
    f.write("{}\n".format(lay_liq))
    f.write("{}\n".format(lay_ice))
    f.close()
    
    for lay in list(np.union1d(lay_liq, lay_ice)):
        LAY = lay
        if LAY in lay_liq and not LAY in lay_ice:
            fracice_lay = 0.0
        elif LAY in lay_ice and not LAY in lay_liq:
            fracice_lay = 1.0
        else:
            fracice_lay = FRACICE
        RECORD_C1_3 += " "
        RECORD_C1_3 += 1 * " " + "{:3d}".format(lay)
        RECORD_C1_3 += "{:>10.5f}".format(CLDFRAC)
        if not homogenous:
            RECORD_C1_3 += "{:>10.5f}".format(CWP[ii])
            RECORD_C1_3 += "{:>10.5f}".format(FRACICE[ii])
            RECORD_C1_3 += "{:>10.5f}".format(EFFSIZEICE[ii])
            RECORD_C1_3 += "{:>10.5f}".format(EFFSIZELIQ[ii])
        else:
            RECORD_C1_3 += "{:>10.5f}".format(CWP/np.union1d(lay_liq, lay_ice).size)
            RECORD_C1_3 += "{:>10.5f}".format(FRACICE)
            RECORD_C1_3 += "{:>10.5f}".format(EFFSIZEICE)
            RECORD_C1_3 += "{:>10.5f}".format(EFFSIZELIQ)
        RECORD_C1_3 += "\n"
        ii += 1
    RECORD_C1_3 += "%"
    
    return RECORD_C1_1 + "\n" + RECORD_C1_3

def create_input_rrtmg_lw(height_prof, press_prof, t_prof, humd_prof, solar_zenith_angle, lat=75., clouds=0, semiss=0.99, co2=None, n2o=None, ch4=None):
    # Record 1.1
    
    CXID = "$ RRTM_LW runscript created on {}".format(dt.datetime.now())
    
    RECORD_1_1 = "{:80s}".format(CXID)
    
    #      RECORD 1.2
     
    #       IAER,    IATM,  IXSECT, NUMANGS,   IOUT,  IDRV,  IMCA, ICLD
     
    #         20,      50,      70,   84-85,  88-90,    92,    94,   95   
     
    #    18x, I2, 29X, I1, 19X, I1, 13X, I2, 2X, I3, 1X,I1,1X, I1,   I1
     
     
    # 	  IAER   (0,10)   flag for aerosols
    #                   = 0   no layers contain aerosols
    #                   = 10  one or more layers contain aerosols (only absorption is treated)
    #                        (requires the presence of file IN_AER_RRTM)
    IAER = 0
    # 	  IATM   (0,1)   flag for RRTATM    1 = yes
    IATM = 1
    # 	  IXSECT (0,1) flag for cross-sections
    #                   = 0  no cross-sections included in calculation
    #                   = 1  cross-sections included in calculation
    IXSECT = 0
    #           NUMANGS = 0  radiance will be computed at 1 angle, the cosine of which
    #                        equals 1/1.66 (standard diffusivity angle approximation)
    #                        (default and only option available)
    NUMANGS = 0
    # 	  IOUT = -1 if no output is to be printed out.
    # 	       =  0 if the only output is for 10-3250 cm-1.
    # 	       =  n (n = 1-16) if the only output is from band n.
    # 		    For the wavenumbers for each band, see Table I.
    # 	       = 99 if output is generated for 17 spectral intervals, one
    # 		    for the full longwave spectrum (10-3250 cm-1), and one 
    # 		    for each of the 16 bands.
    IOUT = 0
    # 	  IDRV   (0,1) flag for applying optional adjustment to the upward flux at
    #                        each layer based on the derivative of the Planck function 
    #                        with respect to temperature for the change in surface 
    #                        temperature (DTBOUND) provided on record 1.4.1 relative to
    #                        the surface temperature (TBOUND) provided on record 1.4
    #                   = 0  standard forward calculation; do not apply derivative 
    #                        adjustment (default)
    #                   = 1  adjust upward flux at each layer based on derivative of 
    #                        the Planck function to account for the change in surface
    # 		       temperature defined by DTBOUND
    IDRV = 0
    # 	  IMCA   (0,1) flag for McICA (Monte Carlo Independent Column Approximation)
    #                        for statistical representation of sub-grid cloud fraction 
    #                        and cloud overlap
    #                   = 0  standard forward calculation; do not use McICA
    #                   = 1  use McICA (will perform statistical sample of 200 forward
    # 		       calculations and output average flux and heating rates)
    if clouds == 0:
        IMCA = 0
    else:
        IMCA = 1
    # 	  ICLD   (0,1,2,3,4,5) flag for clouds
    #                   = 0  no cloudy layers in atmosphere
    #                   = 1  one or more cloudy layers present in atmosphere.  Cloud layers
    # 		       are treated using a RANDOM overlap assmption.
    #                        (requires the presence of file IN_CLD_RRTM for column model)
    #                        (available for IMCA = 0 or 1)
    #                   = 2  one or more cloudy layers present in atmosphere.  Cloud layers
    # 		       are treated using a MAXIMUM/RANDOM overlap assmption.
    #                        (requires the presence of file IN_CLD_RRTM for column model)
    #                        (available for IMCA = 0 or 1)
    #                   = 3  one or more cloudy layers present in atmosphere.  Cloud layers
    # 		       are treated using a MAXIMUM overlap assmption.
    #                        (requires the presence of file IN_CLD_RRTM for column model)
    #                        (available only for IMCA = 1)
    #                   = 4  one or more cloudy layers present in atmosphere.  Cloud layers
    # 		       are treated using a EXPONENTIAL overlap assmption.
    #                        (requires the presence of file IN_CLD_RRTM for column model,
    #                        and inputs set on Records 1.5 and either 1.5.1 or 1.5.2)
    #                        (available only for IMCA = 1)
    #                   = 5  one or more cloudy layers present in atmosphere.  Cloud layers
    # 		       are treated using a EXPONENTIAL-RANDOM overlap assmption.
    #                        (requires the presence of file IN_CLD_RRTM for column model,
    #                        and inputs set on Records 1.5 and either 1.5.1 or 1.5.2)
    #                        (available only for IMCA = 1)
    ICLD = clouds
    
    RECORD_1_2  = 18 * " " + "{:2d}".format(IAER)
    RECORD_1_2 += 29 * " " + "{:1d}".format(IATM)
    RECORD_1_2 += 19 * " " + "{:1d}".format(IXSECT)
    RECORD_1_2 += 13 * " " + "{:2d}".format(NUMANGS)
    RECORD_1_2 += 2  * " " + "{:3d}".format(IOUT)
    RECORD_1_2 += 1  * " " + "{:1d}".format(IDRV)
    RECORD_1_2 += 1  * " " + "{:1d}".format(IMCA)
    RECORD_1_2 += 0  * " " + "{:1d}".format(ICLD)
    
    # RECORD 1.4  
  
    #TBOUND,  IEMIS, IREFLECT, (SEMISS(IB),IB=1,16)
 
    #      1-10,     12,       15,        16-95
 
    #     E10.3  1X, I1,   2X, I1,        16E5.3
 
 
    #     TBOUND   temperature at the surface.  If input value < 0, then surface temperature
    #              is set to the temperature at the bottom of the first layer (see Record 2.1.1)
    TBOUND = t_prof[0]
    #     IEMIS  = 0 each band has surface emissivity equal to 1.0
    #            = 1 each band has the same surface emissivity (equal to SEMISS(16)) 
    #            = 2 each band has different surface emissivity (for band IB, equal to SEMISS(IB))
    IEMIS = 1

    #     IREFLECT = 0 for Lambertian reflection at surface, i.e. reflected radiance 
    #		is equal at all angles
    #              = 1 for specular reflection at surface, i.e. reflected radiance at angle
	#		is equal to downward surface radiance at same angle multiplied by
	#		the reflectance.  THIS OPTION CURRENTLY NOT IMPLEMENTED.
    IREFLECT = 0
    #     SEMISS   the surface emissivity for each band (see Table I).  All values must be 
    #              greater than 0 and less than or equal to 1.  If IEMIS = 1, only
    #              the first value of SEMISS (SEMISS(16)) is considered.  If IEMIS = 2 
    #              and no surface emissivity value is given for SEMISS(IB), a value of 1.0 
    #              is used for band IB.
    SEMISS = [semiss]
    RECORD_1_4  = "{:>10.3f}".format(TBOUND) 
    RECORD_1_4 += 1  * " " + "{:1d}".format(IEMIS)
    RECORD_1_4 += 2  * " " + "{:1d}".format(IREFLECT)
    for element in SEMISS:
        RECORD_1_4 += "{:>5.3f}".format(element)
        
    # ****************************************************************************
    #********     these records applicable if RRTATM selected (IATM=1)    ********
 
    #RECORD 3.1
 
 
    #  MODEL,   IBMAX,  NOPRNT,  NMOL, IPUNCH,   MUNITS,    RE,      CO2MX
 
    #      5,      15,      25,    30,     35,    39-40, 41-50,      71-80

    #     I5,  5X, I5,  5X, I5,    I5,     I5,   3X, I2, F10.3, 20X, F10.3
 
 
    #       MODEL   selects atmospheric profile
 
    #                 = 0  user supplied atmospheric profile
    #                 = 1  tropical model
    #                 = 2  midlatitude summer model
    #                 = 3  midlatitude winter model
    #                 = 4  subarctic summer model
    #                 = 5  subarctic winter model
    #                 = 6  U.S. standard 1976
    MODEL = 0
 
    #       IBMAX     selects layering for RRTM
 
    #                 = 0  RRTM layers are generated internally (default)
    #                 > 0  IBMAX is the number of layer boundaries read in on Record 3.3B which are
    #                             used to define the layers used in RRTM calculation
    IBMAX = len(height_prof)
    #       NOPRNT    = 0  full printout
    #                 = 1  selects short printout
    NOPRINT = 0
    #       NMOL      number of molecular species (default = 7; maximum value is 35)
    NMOL = 7
    #       IPUNCH    = 0  layer data not written (default)
    #                 = 1  layer data written to unit IPU (TAPE7)
    IPUNCH = 1
    #       MUNITS    = 0  write molecular column amounts to TAPE7 (if IPUNCH = 1, default)
    #                 = 1  write molecular mixing ratios to TAPE7 (if IPUNCH = 1)
    MUNITS = 1
    #       RE        radius of earth (km)
	#                defaults for RE=0: 
    #    	        a)  MODEL 0,2,3,6    RE = 6371.23 km
    #			b)        1          RE = 6378.39 km
	#		c)        4,5        RE = 6356.91 km
    RE = 4
    #       CO2MX     mixing ratio for CO2 (ppm).  Default is 330 ppm.
    CO2MX = 400
    
    RECORD_3_1  = "{:5d}".format(MODEL)
    RECORD_3_1 += 5 * " " + "{:5d}".format(IBMAX) 
    RECORD_3_1 += 5 * " " + "{:5d}".format(NOPRINT)
    RECORD_3_1 += "{:5d}".format(NMOL)
    RECORD_3_1 += "{:5d}".format(IPUNCH)
    RECORD_3_1 += 3 * " " + "{:2d}".format(MUNITS)
    RECORD_3_1 += "{:10.3f}".format(RE)

    #   RECORD 3.2
 
 
    #     HBOUND,   HTOA
 
    #       1-10,  11-20
 
    #      F10.3,  F10.3
  
 
    #      HBOUND     altitude of the surface (km)
    HBOUND = height_prof[0]
    
    #      HTOA       altitude of the top of the atmosphere (km)
    HTOA = height_prof[-1]
    
    RECORD_3_2 = "{:10.3f}{:10.3f}".format(HBOUND, HTOA)
    
    # RECORD 3.3B        For IBMAX > 0  (from RECORD 3.1)
 
    #            ZBND(I), I=1, IBMAX   altitudes of RRTM layer boundaries
 
    #           (8F10.3)

 	#            If IBMAX < 0 

	#	PBND(I), I=1, ABS(IBMAX) pressures of LBLRTM layer boundaries

    #           (8F10.3)
    
    ZBND = ""
    for ii in range(len(height_prof)):
    
            ZBND += "{:10.3f}".format(height_prof[ii])
            if ii % 8 == 7:
                ZBND += "\n"
    
    RECORD_3_3B = ZBND
    
    # RECORD 3.4
    
    #           IMMAX,   HMOD
 
    #           5,   6-29
 
    #          I5,    3A8
 
 
    #       IMMAX    number of atmospheric profile boundaries

    #                If IMMAX is set to a negative value, the level boundaries are
    #                specified in PRESSURE (mbars).
    IMMAX = len(height_prof)
    #       HMOD    24 character description of profile
    HMOD = ""
    
    # RECORD 3.5
 
 
    #   ZM,    PM,    TM,    JCHARP, JCHART,   (JCHAR(K),K =1,28)
 
    # 1-10, 11-20, 21-30,        36,     37,     41  through  68
 
    #E10.3, E10.3, E10.3,   5x,  A1,     A1,    3X,    28A1
 
 
    #      ZM       boundary altitude (km). If IMMAX < 0, altitude levels are 
	#	   computed from pressure levels PM. If any altitude levels are
	#	   provided, they are ignored if  IMMAX < 0 (exception: The
	#	   first input level must have an accompanying ZM for input
	#	   into the hydrostatic equation)
    ZM = height_prof
 
    #      PM       pressure (units and input options set by JCHARP)
    PM = press_prof
    
    #      TM       temperature (units and input options set by JCHART)
    TM = t_prof
    
    #  JCHARP       flag for units and input options for pressure (see Table II)
    JCHARP = "A"
    #  JCHART       flag for units and input options for temperature (see Table II)
    JCHART = "A"
    #  JCHAR(K)     flag for units and input options for
    #               the K'th molecule (see Table II)
    #A -> ppmv, B -> cm-3, C -> g/kg, D -> g/m3 (so kann man retrievte Spurengase verwenden) H -> % (relative Humidity)
    # ( 1)  H2O  ( 2)  CO2  ( 3)    O3 ( 4)   N2O ( 5)    CO ( 6)   CH4 ( 7)    O2
    #JCHAR = "H444444"
    JCHAR = "HA4A4A4"
    #JCHAR = "C555555"
     
    #RECORD 3.6.1 ... 3.6.N
 
    #      VMOL(K), K=1, NMOL
 
    #      8E10.3
 
    #      VMOL(K) density of the K'th molecule in units set by JCHAR(K)
    VOL = np.array([np.zeros(len(height_prof)) for ii in range(7)])
    VOL[0] = humd_prof
    VOL[1] = co2
    VOL[2] = np.zeros(len(co2))
    VOL[3] = n2o
    VOL[4] = np.zeros(len(co2))
    VOL[5] = ch4
    VOL[6] = np.zeros(len(co2))
    RECORD_3_5_6 = ""
    for loop in range(len(height_prof)):
        RECORD_3_5_6 += "{:10.3E}".format(ZM[loop])
        RECORD_3_5_6 += "{:10.3E}".format(100*PM[loop])
        RECORD_3_5_6 += "{:10.3f}".format(TM[loop])
        RECORD_3_5_6 += 5 * " " + "{:1s}".format(JCHARP) + "{:1s}".format(JCHART)
        RECORD_3_5_6 += 3 * " " + "{}".format(JCHAR)
        RECORD_3_5_6 += "\n"
        for molecules in range(7):
            RECORD_3_5_6 += "{:10.3E}".format(VOL[molecules, loop])
        RECORD_3_5_6 += "\n"
    # REPEAT records 3.5 and 3.6.1 to 3.6.N for each of the remaining IMMAX boundaries
        
    RECORD_3_4 = "{:5d}".format(IMMAX) + "{:24s}".format(HMOD)
    return RECORD_1_1 + "\n" + RECORD_1_2 + "\n" + RECORD_1_4 + "\n" + \
            RECORD_3_1 + "\n" + RECORD_3_2 + "\n" + RECORD_3_3B + "\n" + RECORD_3_4 + "\n" + \
            RECORD_3_5_6



def create_input_rrtmg_sw(height_prof, press_prof, t_prof, humd_prof, solar_zenith_angle, lat=75., clouds=0, albedo_dir=0.3, albedo_diff=0.3, co2=None, n2o=None, ch4=None):
    # Record 1.1
    
    CXID = "$ RRTM_SW runscript created on {}".format(dt.datetime.now())
    
    RECORD_1_1 = "{:80s}".format(CXID)
    
    # RECORD 1.2
    # 
    #      IAER,    IATM,   ISCAT,  ISTRM,   IOUT,   ICLD,  IDELM, ICOS
    # 
    #        20,      50,      83,     85,  88-90,     95,     99,  100
    # 
    #   18x, I2, 29X, I1, 32X, I1, 1X, I1, 2X, I3, 4X, I1, 3X, I1,   I1
     
        
    
    #	  IAER   (0,10)   flag for aerosols
    #                  = 0   no layers contain aerosols
    #                  = 10  one or more layers contain aerosols
    #                       (requires the presence of file IN_AER_RRTM)
    IAER = 0 
    
    #	  IATM   (0,1)   flag for RRTATM    1 = yes
    IATM = 1
    
    #         ISCAT  (0,1) switch for DISORT or simple two-stream scattering 
    #                 = 0  DISORT (unavailable)
    #                 = 1  two-stream    (default)
    ISCAT = 1
    
    #          ISTRM   flag for number of streams used in DISORT  (ISCAT must be equal to 0)
    #                  = 0  - 4 streams (default)
    #                  = 1  - 8 streams
    #                  = 2  - 16 streams 
    ISTRM = 2
    
    #	  IOUT    = -1 if no output is to be printed out.
    #	          =  0 if the only output is for 820-50000 cm-1.
    #	          =  n (n = 16-29) if the only output is from band n.
    #		       For the wavenumbers for each band, see Table I.
    #	          = 98 if output is generated for 15 spectral intervals, one
    #		       for the full shortwave spectrum (820-50000 cm-1), and one 
    #		       for each of the 14 bands.
    IOUT = 0
    
    if clouds == 0:
        ICMA = 0
    else:
        ICMA = 1
    
    # 	  ICLD   (0,1) flag for clouds   (currently not implemented)
    #                  = 0  no cloudy layers in atmosphere
    #                  = 1  one or more cloudy layers present in atmosphere
    #                       (requires the presence of file IN_CLD_RRTM)
    ICLD = clouds
    
    #  Measurement comparison flags:
    #          IDELM  (0,1) flag for outputting downwelling fluxes computed using the delta-M scaling approximation
    #                  = 0  output "true" direct and diffuse downwelling fluxes
    #                  = 1  output direct and diffuse downwelling fluxes computed with delta-M approximation
    #                 (Note:  The delta-M approximation is always used internally in RRTM_SW to compute the total
    #                 downwelling flux at each level.  What the IDELM flag determines is whether the components 
    #                 of the downwelling flux, the direct and diffuse fluxes, that are output are the actual direct
    #                 and diffuse fluxes (IDELM = 0) or are those computed using the delta-M approximation (IDELM = 1).
    #                 If the computed direct and diffuse fluxes are being compared with corresponding measured fluxes 
    #                 and a nontrivial amount of forward scattered radiation is likely to have been included in the
    #                 measurement of the direct flux, then IDELM should be set to 1.)
    IDELM = 0
    
    #          ICOS   = 0 there is no need to account for instrumental cosine response
    #                 = 1 to account for instrumental cosine response in the computation of the direct and diffuse fluxes 
    #                 = 2 to account for instrumental cosine response in the computation of the diffuse fluxes only
    #                 (Note:  ICOS = 1 and ICOS = 2 requires the presence of the file COSINE_RESPONSE, which should 
    #                 consist of lines containing pairs of numbers (ANG, COSFAC), where COSFAC is the instrumental cosine
    #                 response at the angle ANG.) 
    ICOS = 0
    
    RECORD_1_2  = 18 * " " + "{:2d}".format(IAER)
    RECORD_1_2 += 29 * " " + "{:1d}".format(IATM)
    RECORD_1_2 += 32 * " " + "{:1d}".format(ISCAT)
    RECORD_1_2 += 1  * " " + " "#"{:1d}".format(ISTRM)
    RECORD_1_2 += 2  * " " + "{:3d}".format(IOUT)
    RECORD_1_2 += 3  * " " + "{:1d}".format(ICMA)
    RECORD_1_2 += 0  * " " + "{:1d}".format(ICLD)
    RECORD_1_2 += 3  * " " + "{:1d}".format(IDELM)
    RECORD_1_2 += "{:1d}".format(ICOS)
                       
    # RECORD 1.2.1  

    #       JULDAT,      SZA, ISOLVAR,   SCON, SOLCYCFRAC, (SOLVAR(IB),IB=16,29)

    #        13-15,    19-25,   29-30,  31-40,      41-50,   51-190

    #    12X,   I3, 3X, F7.4,  3X, I2,  F10.4,      F10.5,  14F10.5

	
    #    JULDAT       Julian day associated with calculation (1-365/366 starting January 1).
	#             Used to calculate Earth distance from sun. A value of 0 (default) indicates 
	#             no scaling of solar source function using earth-sun distance.
    JULDAT = 0#0.3907103825136612
    #    SZA          Solar zenith angle in degrees (0 degrees is overhead).

    mu_sza = np.cos(solar_zenith_angle * np.pi / 180.0)
    const = 0.001277
    mu_sza_adj = const / (np.sqrt((mu_sza)**2 + const * 2 + const**2) - mu_sza)
    sza_adj = np.arccos(mu_sza_adj) * 180 / np.pi
    SZA = sza_adj#solar_zenith_angle
    #    ISOLVAR      Solar variability option [-1,0,1,2,3]

	#	     =-1 (when SCON .EQ. 0.0): no solar variability; each band uses the Kurucz 
    #                     extraterrestrial solar irradiance, corresponding to a spectrally integrated 
    #                     solar constant of 1368.22 Wm-2 (method used in rrtmg_sw_v3.91 and earlier)
    #                 =-1 (when SCON .NE. 0.0): solar variability active; baseline solar irradiance
    #                     of 1368.22 Wm-2 is scaled to SCON, solar variability is determined (optional)
    #                     by non-zero scale factors for each band defined by SOLVAR

	#	     = 0 (when SCON .EQ. 0.0): no solar variability; each band uses the solar constant 
    #                     from the NRLSSI2 model of 1360.85 Wm-2 (for the spectral range 100-50000 cm-1) 
    #                     with quiet sun, facular and sunspot contributions fixed to the mean of 
    #                     Solar Cycles 13-24 and averaged over the mean solar cycle
	#	     = 0 (when SCON .NE. 0.0): no solar variability; baseline solar irradiance of
    #                     1360.85 Wm-2 (for the spectral range 100-50000 cm-1) is scaled to SCON

	#	     = 1 solar variability active; solar cycle contribution determined by input of
    #                     SOLCYCFRAC, a fraction representing the phase of the solar cycle, with 
    #                     facular brightening and sunspot blocking effects varying in time with this
    #                     fraction through their mean variations over the average of Solar Cycles 13-24
    #                     (corresponding to a solar constant of 1360.85 Wm-2); two amplitude scale 
    #                     factors provided in SOLVAR allow independent adjustment of facular and sunspot 
    #                     effects from their mean solar cycle amplitudes

	#	     = 2 solar variability active; solar cycle contribution determined by direct
	# 	         specification of Mg (facular) and SB (sunspot) indices consistent with the
    #                     NRLSSI2 solar model; these are provided in SOLVAR and are used to model the 
    #                     solar variability at a specific time for a specific solar cycle 
    #                     (SCON = 0.0 only; solar constant depends on Mg and SB indices provided)
    #                     Further information on setting the Mg and SB indices for this option can 
    #                     be found at the NRLSSI model github site: https://github.com/lasp/nrlssi.

	#	     = 3 (when SCON .EQ. 0.0): no solar variability; each band uses the NRLSSI2
    #                     extraterrestrial solar irradiance, corresponding to a spectrally integrated 
    #                     solar constant of 1360.85 Wm-2 with quiet sun, facular and sunspot
    #                     contributions averaged over the mean of Solar Cycles 13-24
    #                 = 3 (when SCON .NE. 0.0): solar variability active; baseline solar irradiance
    #                     of 1360.85 Wm-2 is scaled to SCON, solar variability is determined (optional)
    #                     by non-zero scale factors for each band defined by SOLVAR and applied to SCON
    ISOLVAR = 0
    #    SCON         For ISOLVAR = -1 or 0:
    #                     Total solar irradiance (if SCON > 0, internal solar irradiance is scaled 
    #                     to this value)
    #                 For ISOLVAR = 1:
    #                     Solar constant; integral of total solar irradiance averaged over solar cycle
    #                     (if SCON > 0, internal solar irradiance is scaled to this value)
    #                 For ISOLVAR = 2:
    #                     SCON must be 0.0, since total solar irradiance is defined by the Mg and SB
    #                     indices provided in SOLVAR
    #                 For ISOLVAR = 3:
    #                     Total solar irradiance before individual band scale factors are applied
    #                     (if SCON > 0 internal solar irradiance is scaled to this value)
    #                 Set SCON = 0.0 to use internal solar irradiance, which depends on ISOLVAR
    SCON = 0
    #    SOLCYCFRAC   Solar cycle fraction (0-1); fraction of the way through the mean 11-year
    #                 cycle with 0.0 defined as the first day of year 1 and 1.0 defined as the
    #                 last day of year 11 (ISOLVAR=1 only). Note that for the combined effect of
    #                 the solar constant of 1360.85, and the mean facular brightening and sunspot 
    #                 dimming components (without scaling), the minimum total solar irradiance of
    #                 1360.49 occurs at SOLCYCFRAC = 0.0265, and the maximum total solar irradiance 
    #                 of 1361.34 occurs at SOLCYCFRAC = 0.3826. 
    SOLCYCFRAC = 0
	#SOLVAR       Solar variability scaling factors or indices (ISOLVAR=-1,1,2,3 only)
	#             For ISOLVAR = 1:
    #                   SOLVAR(1)    Facular (Mg) index amplitude scale factor
    #                   SOLVAR(2)    Sunspot (SB) index amplitude scale factor

	#             For ISOLVAR = 2:
    #                   SOLVAR(1)    Facular (Mg) index as defined in the NRLSSI2 model;
    #                                used for modeling time-specific solar activity
    #                   SOLVAR(2)    Sunspot (SB) index as defined in the NRLSSI2 model; 
    #                                used for modeling time-specific solar activity

	#             For ISOLVAR = -1 or 3:
    #                   SOLVAR(1:14) Band scale factors for modeling spectral variation of 
    #                                averaged solar cycle in each shortwave band
    SOLVAR = 0
    RECORD_1_2_1  = 12 * " " + "{:3d}".format(JULDAT)
    RECORD_1_2_1 += 3  * " " + "{:7.4f}".format(SZA)
    RECORD_1_2_1 += 3  * " " + "{:2d}".format(ISOLVAR)
    RECORD_1_2_1 += "{:10.4f}".format(SCON)
    RECORD_1_2_1 += "{:10.5f}".format(SOLCYCFRAC)
    if ISOLVAR > 0:
        for ii in range(14):
            RECORD_1_2_1 += "{:5.3f}".format(SOLVAR[ii])
            
    # RECORD 1.4  
  
    #       IEMIS, IREFLECT, (SEMISS(IB),IB=16,29)
 
    #          12,       15,                 16-85
 
    #     11X, I1,   2X, I1,                14F5.3
 
    #    (Note:  surface reflectance = 1 - surface emissivity) 

    #     IEMIS  = 0 each band has surface emissivity equal to 1.0
    #            = 1 each band has the same surface emissivity (equal to SEMISS(16)) 
    #            = 2 each band has different surface emissivity (for band IB, equal to SEMISS(IB))
    IEMIS = 1

    #     IREFLECT = 0 for Lambertian reflection at surface, i.e. reflected radiance 
    #		is equal at all angles
    #              = 1 for specular reflection at surface, i.e. reflected radiance at angle
	#		is equal to downward surface radiance at same angle multiplied by
	#		the reflectance.  THIS OPTION CURRENTLY NOT IMPLEMENTED.
    IREFLECT = 0
    #     SEMISS   the surface emissivity for each band (see Table I).  All values must be 
    #              greater than 0 and less than or equal to 1.  If IEMIS = 1, only
    #              the first value of SEMISS (SEMISS(16)) is considered.  If IEMIS = 2 
    #              and no surface emissivity value is given for SEMISS(IB), a value of 1.0 
    #              is used for band IB.
    SEMISS_DIF = [1-albedo_dir]#[0.8]
    SEMISS_DIR = [1-albedo_diff]
    
    RECORD_1_4  = 11 * " " + "{:1d}".format(IEMIS)
    RECORD_1_4 += 2  * " " + "{:1d}".format(IREFLECT)
    for element in SEMISS_DIR:
        RECORD_1_4 += "{:5.3f}".format(element)
    RECORD_1_4 += "\n"
    RECORD_1_4 +=15*' '
    for element in SEMISS_DIF:
        RECORD_1_4 += "{:5.3f}".format(element)
        
    # ****************************************************************************
    #********     these records applicable if RRTATM selected (IATM=1)    ********
 
    #RECORD 3.1
 
 
    #  MODEL,   IBMAX,  NOPRNT,  NMOL, IPUNCH,   MUNITS,    RE,      CO2MX, REF_LAT
 
    #      5,      15,      25,    30,     35,    39-40, 41-50,      71-80, 81-90

    #     I5,  5X, I5,  5X, I5,    I5,     I5,   3X, I2, F10.3, 20X, F10.3, F10.3
 
 
    #       MODEL   selects atmospheric profile
 
    #                 = 0  user supplied atmospheric profile
    #                 = 1  tropical model
    #                 = 2  midlatitude summer model
    #                 = 3  midlatitude winter model
    #                 = 4  subarctic summer model
    #                 = 5  subarctic winter model
    #                 = 6  U.S. standard 1976
    MODEL = 0
 
    #       IBMAX     selects layering for RRTM
 
    #                 = 0  RRTM layers are generated internally (default)
    #                 > 0  IBMAX is the number of layer boundaries read in on Record 3.3B which are
    #                             used to define the layers used in RRTM calculation
    IBMAX = len(height_prof)
    #       NOPRNT    = 0  full printout
    #                 = 1  selects short printout
    NOPRINT = 1
    #       NMOL      number of molecular species (default = 7; maximum value is 35)
    NMOL = 7
    #       IPUNCH    = 0  layer data not written (default)
    #                 = 1  layer data written to unit IPU (TAPE7)
    IPUNCH = 1
    #       MUNITS    = 0  write molecular column amounts to TAPE7 (if IPUNCH = 1, default)
    #                 = 1  write molecular mixing ratios to TAPE7 (if IPUNCH = 1)
    MUNITS = 1
    #       RE        radius of earth (km)
	#                defaults for RE=0: 
    #    	        a)  MODEL 0,2,3,6    RE = 6371.23 km
    #			b)        1          RE = 6378.39 km
	#		c)        4,5        RE = 6356.91 km
    RE = 4
    #       CO2MX     mixing ratio for CO2 (ppm).  Default is 330 ppm.
    CO2MX = 400
	# REF_LAT     latitude of location of calculation (degrees)
	#	     defaults for REF_LAT = 0:
	#	     a) MODEL 0,2,3,6    REF_LAT = 45.0 degrees
	#	     b) MODEL 1          REF_LAT = 15.0
	#	     c) MODEL 4,5        REF_LAT = 60.0
    REF_LAT = lat
    
    RECORD_3_1  = "{:5d}".format(MODEL)
    RECORD_3_1 += 5 * " " + "{:5d}".format(IBMAX) 
    RECORD_3_1 += 5 * " " + "{:5d}".format(NOPRINT)
    RECORD_3_1 += "{:5d}".format(NMOL)
    RECORD_3_1 += "{:5d}".format(IPUNCH)
    RECORD_3_1 += 3 * " " + "{:2d}".format(MUNITS)
    RECORD_3_1 += "{:10.3f}".format(RE)
    RECORD_3_1 += 20 * " " + "{:10.3f}".format(CO2MX)
    RECORD_3_1 += "{:10.3f}".format(REF_LAT)
    
    #   RECORD 3.2
 
 
    #     HBOUND,   HTOA
 
    #       1-10,  11-20
 
    #      F10.3,  F10.3
  
 
    #      HBOUND     altitude of the surface (km)
    HBOUND = height_prof[0]
    
    #      HTOA       altitude of the top of the atmosphere (km)
    HTOA = height_prof[-1]
    
    RECORD_3_2 = "{:10.3f}{:10.3f}".format(HBOUND, HTOA)
    
    # RECORD 3.3B        For IBMAX > 0  (from RECORD 3.1)
 
    #            ZBND(I), I=1, IBMAX   altitudes of RRTM layer boundaries
 
    #           (8F10.3)

 	#            If IBMAX < 0 

	#	PBND(I), I=1, ABS(IBMAX) pressures of LBLRTM layer boundaries

    #           (8F10.3)
    
    ZBND = ""
    for ii in range(len(height_prof)):
    
            ZBND += "{:10.3f}".format(height_prof[ii])
            if ii % 8 == 7:
                ZBND += "\n"
    
    RECORD_3_3B = ZBND
    
    # RECORD 3.4
    
    #           IMMAX,   HMOD
 
    #           5,   6-29
 
    #          I5,    3A8
 
 
    #       IMMAX    number of atmospheric profile boundaries

    #                If IMMAX is set to a negative value, the level boundaries are
    #                specified in PRESSURE (mbars).
    IMMAX = len(height_prof)
    #       HMOD    24 character description of profile
    HMOD = ""
    
    # RECORD 3.5
 
 
    #   ZM,    PM,    TM,    JCHARP, JCHART,   (JCHAR(K),K =1,28)
 
    # 1-10, 11-20, 21-30,        36,     37,     41  through  68
 
    #E10.3, E10.3, E10.3,   5x,  A1,     A1,    3X,    28A1
 
 
    #      ZM       boundary altitude (km). If IMMAX < 0, altitude levels are 
	#	   computed from pressure levels PM. If any altitude levels are
	#	   provided, they are ignored if  IMMAX < 0 (exception: The
	#	   first input level must have an accompanying ZM for input
	#	   into the hydrostatic equation)
    ZM = height_prof
 
    #      PM       pressure (units and input options set by JCHARP)
    PM = press_prof
    
    #      TM       temperature (units and input options set by JCHART)
    TM = t_prof
    
    #  JCHARP       flag for units and input options for pressure (see Table II)
    JCHARP = "A"
    #  JCHART       flag for units and input options for temperature (see Table II)
    JCHART = "A"
    #  JCHAR(K)     flag for units and input options for
    #               the K'th molecule (see Table II)
    #A -> ppmv, B -> cm-3, C -> g/kg, D -> g/m3 (so kann man retrievte Spurengase verwenden)
    # ( 1)  H2O  ( 2)  CO2  ( 3)    O3 ( 4)   N2O ( 5)    CO ( 6)   CH4 ( 7)    O2
    #JCHAR = "H444444"
    JCHAR = "HA4A4A4"
    #JCHAR = "H111111"
     
    #RECORD 3.6.1 ... 3.6.N
 
    #      VMOL(K), K=1, NMOL
 
    #      8E10.3
 
    #      VMOL(K) density of the K'th molecule in units set by JCHAR(K)
    VOL = np.array([np.zeros(len(height_prof)) for ii in range(7)])
    VOL[0] = humd_prof
    VOL[1] = co2
    VOL[2] = np.zeros(len(co2))
    VOL[3] = n2o
    VOL[4] = np.zeros(len(co2))
    VOL[5] = ch4
    VOL[6] = np.zeros(len(co2))
    RECORD_3_5_6 = ""
    for loop in range(len(height_prof)):
        RECORD_3_5_6 += "{:10.3E}".format(ZM[loop])
        RECORD_3_5_6 += "{:10.3E}".format(100*PM[loop])
        RECORD_3_5_6 += "{:10.3E}".format(TM[loop])
        RECORD_3_5_6 += 5 * " " + "{:1s}".format(JCHARP) + "{:1s}".format(JCHART)
        RECORD_3_5_6 += 3 * " " + "{}".format(JCHAR)
        RECORD_3_5_6 += "\n"
        for molecules in range(7):
            RECORD_3_5_6 += "{:10.3E}".format(VOL[molecules, loop])
        RECORD_3_5_6 += "\n"
    # REPEAT records 3.5 and 3.6.1 to 3.6.N for each of the remaining IMMAX boundaries
    
    RECORD_3_4 = "{:5d}".format(IMMAX) + "{:24s}".format(HMOD)
    return RECORD_1_1 + "\n" + RECORD_1_2 + "\n" + RECORD_1_2_1 + "\n" + RECORD_1_4 + "\n" + \
            RECORD_3_1 + "\n" + RECORD_3_2 + "\n" + RECORD_3_3B + "\n" + RECORD_3_4 + "\n" + \
            RECORD_3_5_6

'''
def read_results(layer, spec_range, keys, fname="OUTPUT_RRTM"):


    if spec_range == 'sw_sum':
        WINDOWS = ['820. - 50000.']
        add = 3
    elif spec_range == 'lw_sum':
        WINDOWS = ['10.0 - 3250.0']
        add = 2
        
    with open(fname, "r") as f:
        cont = f.readlines()
    for ii in range(len(cont)):
        for win in range(len(WINDOWS)):
            if WINDOWS[win] in cont[ii]:
                line = []
                for kk in range(layer+1):
                    dummy = np.array([])
                    for jj in range(len(cont[ii+kk+add].split(" "))):
                        print(ii+kk+add)
                        print(cont[ii+kk+add])
                        try:
                            if "*" in cont[ii+kk+add].split(" ")[jj]:
                                dummy = np.concatenate((dummy, [0]))
                            else:
                                dummy = np.concatenate((dummy, [float(cont[ii+kk+add].split(" ")[jj])]))
                        except FileNotFoundError:
                            pass
                    line.append(dummy)
                line = np.array(line[1:])

                out = dict()
                for key in range(len(keys)):
                    out.update({keys[key]: line[:, key]})
                out = pd.DataFrame(out)

                break
    return out
'''
def read_results(layer, spec_range, keys, fname="OUTPUT_RRTM"):
    with open(fname, "r") as f:
        cont = f.readlines()
        
    header = []
    for line in cont:
        #print(line)
        
        ## find header
        if "LEVEL" in line:
            for element in line.split("  "):
                if len(element) > 0:
                    if element.rstrip().lstrip() == "LEVEL PRESSURE":
                        text = "LEVEL"
                        header.append(text)
                        text = "PRESSURE"
                        header.append(text)
                    else:
                        text = element.rstrip().lstrip()
                        header.append(text)
                        
     
    for ii in range(len(cont)):
        if "LEVEL" in cont[ii]:
            start = ii+2
        if " 0 " in cont[ii]:
            end = ii+1


    col = [[] for ii in range(len(header))]

    for line in cont[start:end]:
        ii = 0
        for element in line.split(" "):
            if len(element) > 0:
                if "*" in element:
                    col[ii].append(0)
                else:
                    col[ii].append(float(element))
                ii += 1
    col = np.array(col)
    out = dict()
    for key in range(len(header)):
        out.update({header[key]: col[key, :]})
    out = pd.DataFrame(out)
    return out

def calc_clear_sky(z, p, t, q, sza, albedo_dir, albedo_diff, semiss, lat, co2, n2o, ch4):
    ret = create_input_rrtmg_sw(height_prof=z, \
                           press_prof=p*1e-2, \
                           t_prof=t, \
                           humd_prof=q, \
                           solar_zenith_angle=sza, \
                           clouds=0, \
                          albedo_dir=albedo_dir,\
                              albedo_diff=albedo_diff, \
                               lat=lat, co2=co2, n2o=n2o, ch4=ch4)
    
    with open("INPUT_RRTM", "w") as f:
        f.write(ret)
    #subprocess.call(['cp', 'INPUT_RRTM', 'INPUT_RRTM_SW'])
    
       
    subprocess.call(['{}'.format(SRC_RRTMG_SW)])
    clear_sw = read_results(len(z), 'sw_sum', KEYS_SW)
    print(clear_sw)
    subprocess.call(['mv', 'OUTPUT_RRTM', 'OUTPUT_RRTM_SW_CLEAR'])

    ret = create_input_rrtmg_lw(height_prof=z, \
                           press_prof=p*1e-2, \
                           t_prof=t, \
                           humd_prof=q, \
                           solar_zenith_angle=sza, \
                           clouds=0, \
                           semiss=semiss, co2=co2, n2o=n2o, ch4=ch4)
    
    with open("INPUT_RRTM", "w") as f:
        f.write(ret)
    #subprocess.call(['cp', 'INPUT_RRTM', 'INPUT_RRTM_LW'])

        
    subprocess.call(['{}'.format(SRC_RRTMG_LW)])
    clear_lw = read_results(len(z), 'lw_sum', KEYS_LW)
    subprocess.call(['mv', 'OUTPUT_RRTM', 'OUTPUT_RRTM_LW_CLEAR'])
    return clear_sw, clear_lw

def calc_all_sky(z, p, t, q, sza, albedo_dir, albedo_diff, cloud, cwp, rl, ri, wpi, semiss, lat, co2, n2o, ch4, clt):
    ret = create_input_rrtmg_sw(height_prof=z, \
                           press_prof=p*1e-2, \
                           t_prof=t, \
                           humd_prof=q, \
                           solar_zenith_angle=sza, \
                           clouds=2, \
                           albedo_dir=albedo_dir, \
                               albedo_diff=albedo_diff, \
                               lat=lat, co2=co2, n2o=n2o, ch4=ch4)
    cld = create_cloud_rrtmg(lay_liq=cloud, lay_ice=cloud, cwp=cwp, rliq=rl, rice=ri, fice=wpi, homogenous=False, clt=clt)
    with open("INPUT_RRTM", "w") as f:
        f.write(ret)
    #subprocess.call(['cp', 'INPUT_RRTM', 'INPUT_RRTM_SW'])
    with open("IN_CLD_RRTM", "w") as f:
        f.write(cld)  
    subprocess.call(['{}'.format(SRC_RRTMG_SW)])
    all_sw = read_results(len(z), 'sw_sum', KEYS_SW)

    subprocess.call(['cp', 'OUTPUT_RRTM', 'OUTPUT_RRTM_SW_ALL'])
    
    ret = create_input_rrtmg_lw(height_prof=z, \
                       press_prof=p*1e-2, \
                       t_prof=t, \
                       humd_prof=q, \
                       solar_zenith_angle=sza, \
                       clouds=2, \
                       semiss=semiss, co2=co2, n2o=n2o, ch4=ch4)
        
    with open("INPUT_RRTM", "w") as f:
        f.write(ret)
    #subprocess.call(['cp', 'INPUT_RRTM', 'INPUT_RRTM_LW'])


    subprocess.call(['{}'.format(SRC_RRTMG_LW)])
    all_lw = read_results(len(z), 'lw_sum', KEYS_LW)
    subprocess.call(['cp', 'OUTPUT_RRTM', 'OUTPUT_RRTM_LW_ALL'])
    
    return all_sw, all_lw
    
def error_propagation(dataframe_err, dataframe, delta):
    '''
    Brief
    -----
    Calculate difference quotient

    Parameters
    ----------
    dataframe_err : Pandas.DataFrame
        Dataframe containing disturbed calculations
    dataframe : TYPE
        Dataframe containing the raw calculations
    delta : float
        h (https://en.wikipedia.org/wiki/Difference_quotient)

    Returns
    -------
    Pandas.DataFrame
        Difference quotient

    '''
    return (dataframe_err - dataframe)/delta    

def write_results(all_sw, all_lw, clear_sw, clear_lw,  deriv_cwp_lw, deriv_cwp_sw, deriv_rl_lw, deriv_rl_sw, deriv_ri_lw, deriv_ri_sw, deriv_wpi_lw, deriv_wpi_sw, lat, lon, sza, cwp, wpi, rl, ri, cloud, z, t, q, p, dcwp, dwpi, drl, dri, co2, n2o, ch4, albedo_dir, albedo_diff, iceconc, semiss, clt, fname):

    with netcdf_file(fname, "w") as outfile:
        outfile.createDimension("const", 1)
        outfile.createDimension("level", z.size)
        outfile.createDimension('cgrid', cloud.size)
        
        height_out = outfile.createVariable("height", "f8", ("level", ))
        height_out.units = "km"
        height_out[:] = z[:]
        
        pressure_out = outfile.createVariable("pressure", "f8", ("level", ))
        pressure_out.units = "hPa"
        pressure_out[:] = p[:]
        
        temperature_out = outfile.createVariable("temperature", "f8", ("level", ))
        temperature_out.units = "K"
        temperature_out[:] = t[:]
        
        humidity_out = outfile.createVariable("humidity", "f8", ("level", ))
        humidity_out.units = "%"
        humidity_out[:] = q[:]
        
        lat_out = outfile.createVariable("latitude", "f8", ("const", ))
        lat_out.units = "DegN"
        lat_out[:] = lat

        lon_out = outfile.createVariable("longitude", "f8", ("const", ))
        lon_out.units = "DegE"
        lon_out[:] = lon
        
        sza_out = outfile.createVariable("solar_zenith_angle", "f8", ("const", ))
        sza_out.units = "Deg"
        sza_out[:] = sza
        
        sic_out = outfile.createVariable("sea_ice_concentration", "f8", ("const", ))
        sic_out.units = "1"
        sic_out[:] = iceconc
        
        albedo_dir_out = outfile.createVariable("sw_broadband_surface_albedo_direct_radiation", "f8", ("const", ))
        albedo_dir_out.units = "1"
        albedo_dir_out[:] = albedo_dir
        
        albedo_diff_out = outfile.createVariable("sw_broadband_surface_albedo_diffuse_radiation", "f8", ("const", ))
        albedo_diff_out.units = "1"
        albedo_diff_out[:] = albedo_diff
        
        semiss_out = outfile.createVariable("lw_surface_emissivity", "f8", ("const", ))
        semiss_out.units = "1"
        semiss_out[:] = semiss
        
        cloud_out = outfile.createVariable("cloud_idx", "i4", ("cgrid", ))
        cloud_out[:] = cloud
        
        cwp_out = outfile.createVariable("CWP", "f8", ("cgrid", ))
        cwp_out.units = "gm-2"
        cwp_out[:] = cwp
        
        dcwp_out = outfile.createVariable("delta_CWP", "f8", ("cgrid", ))
        dcwp_out.units = "gm-2"
        dcwp_out[:] = dcwp

        rl_out = outfile.createVariable("rl", "f8", ("cgrid", ))
        rl_out.units = "um"
        rl_out[:] = rl
        
        drl_out = outfile.createVariable("delta_rl", "f8", ("cgrid", ))
        drl_out.units = "um"
        drl_out[:] = drl

        ri_out = outfile.createVariable("ri", "f8", ("cgrid", ))
        ri_out.units = "um"
        ri_out[:] = ri
        
        dri_out = outfile.createVariable("delta_ri", "f8", ("cgrid", ))
        dri_out.units = "um"
        dri_out[:] = dri

        wpi_out = outfile.createVariable("WPi", "f8", ("cgrid", ))
        wpi_out.units = "1"
        wpi_out[:] = wpi
        
        dwpi_out = outfile.createVariable("delta_WPi", "f8", ("cgrid", ))
        dwpi_out.units = "1"
        dwpi_out[:] = dwpi
        
        clt_out = outfile.createVariable("cloud_fraction", "f8", ("const", ))
        clt_out.units = "1"
        clt_out[:] = clt
        
        co2_out = outfile.createVariable("co2_profile", "f8", ("level", ))
        co2_out.units = "ppmv"
        co2_out[:] = co2
        
        n2o_out = outfile.createVariable("n2o_profile", "f8", ("level", ))
        n2o_out.units = "ppmv"
        n2o_out[:] = n2o

        ch4_out = outfile.createVariable("ch4_profile", "f8", ("level", ))
        ch4_out.units = "ppmv"
        ch4_out[:] = ch4
        
        oob_liq = np.where((rl < 2.5) & \
                           (wpi < 1.0))[0].size
        oob_liq+= np.where((rl >60.0) & \
                           (wpi < 1.0))[0].size
        oob_ice = np.where((ri < 13.0) & \
                           (wpi > 0.0))[0].size
        oob_ice+= np.where((ri > 131.0) & \
                           (wpi > 0.0))[0].size
                
        oob = outfile.createVariable("out_of_bounds", "i4", ("const", ))
        oob[:] = oob_liq + oob_ice
        
        clear_sw_out = [None for ii in range(len(clear_sw.keys()))]
        all_sw_out = [None for ii in range(len(all_sw.keys()))]
        deriv_cwp_sw_out = [None for ii in range(len(all_sw.keys()))]
        deriv_rl_sw_out = [None for ii in range(len(all_sw.keys()))]
        deriv_ri_sw_out = [None for ii in range(len(all_sw.keys()))]
        deriv_wpi_sw_out = [None for ii in range(len(all_sw.keys()))]
        for ii in range(len(all_sw.keys())):
            if all_sw.keys()[ii] == "LEVEL" or all_sw.keys()[ii] == "PRESSURE":
                continue
            clear_sw_out[ii] = outfile.createVariable("clear_sw_{}".format(clear_sw.keys()[ii]), 'f8', ('level', ))
            all_sw_out[ii] = outfile.createVariable("all_sw_{}".format(all_sw.keys()[ii]), 'f8', ('level', ))
            deriv_cwp_sw_out[ii] = outfile.createVariable("difference_quotient_cwp_sw_{}".format(all_sw.keys()[ii]), 'f8', ('level', ))
            deriv_rl_sw_out[ii] = outfile.createVariable("difference_quotient_rl_sw_{}".format(all_sw.keys()[ii]), 'f8', ('level', ))
            deriv_ri_sw_out[ii] = outfile.createVariable("difference_quotient_ri_sw_{}".format(all_sw.keys()[ii]), 'f8', ('level', ))
            deriv_wpi_sw_out[ii] = outfile.createVariable("difference_quotient_wpi_sw_{}".format(all_sw.keys()[ii]), 'f8', ('level', ))

            if all_sw.keys()[ii] == "HEATING RATE":
                all_sw_out[ii].units = "degree/day"
                clear_sw_out[ii].units = "degree/day"
                deriv_cwp_sw_out[ii].units = "degree/day"
                deriv_rl_sw_out[ii].units = "degree/day"
                deriv_ri_sw_out[ii].units = "degree/day"
                deriv_wpi_sw_out[ii].units = "degree/day"

            else:
                all_sw_out[ii].units = "Wm-2"
                clear_sw_out[ii].units = "Wm-2"
                deriv_cwp_sw_out[ii].units = "Wm-2"
                deriv_rl_sw_out[ii].units = "Wm-2"
                deriv_ri_sw_out[ii].units = "Wm-2"
                deriv_wpi_sw_out[ii].units = "Wm-2"

            all_sw_out[ii][:] = all_sw[all_sw.keys()[ii]][::-1]
            clear_sw_out[ii][:] = clear_sw[clear_sw.keys()[ii]][::-1]
            deriv_cwp_sw_out[ii][:] = deriv_cwp_sw[all_sw.keys()[ii]][::-1]
            deriv_rl_sw_out[ii][:] = deriv_rl_sw[all_sw.keys()[ii]][::-1]
            deriv_ri_sw_out[ii][:] = deriv_ri_sw[all_sw.keys()[ii]][::-1]
            deriv_wpi_sw_out[ii][:] = deriv_wpi_sw[all_sw.keys()[ii]][::-1]
            
        clear_lw_out = [None for ii in range(len(clear_lw.keys()))]
        all_lw_out = [None for ii in range(len(all_lw.keys()))]
        deriv_cwp_lw_out = [None for ii in range(len(all_lw.keys()))]
        deriv_rl_lw_out = [None for ii in range(len(all_lw.keys()))]
        deriv_ri_lw_out = [None for ii in range(len(all_lw.keys()))]
        deriv_wpi_lw_out = [None for ii in range(len(all_lw.keys()))]
        for ii in range(len(all_lw.keys())):
            if all_lw.keys()[ii] == "LEVEL" or all_lw.keys()[ii] == "PRESSURE":
                continue
            clear_lw_out[ii] = outfile.createVariable("clear_lw_{}".format(clear_lw.keys()[ii]), 'f8', ('level', ))
            all_lw_out[ii] = outfile.createVariable("all_lw_{}".format(all_lw.keys()[ii]), 'f8', ('level', ))
            deriv_cwp_lw_out[ii] = outfile.createVariable("difference_quotient_cwp_lw_{}".format(all_lw.keys()[ii]), 'f8', ('level', ))
            deriv_rl_lw_out[ii] = outfile.createVariable("difference_quotient_rl_lw_{}".format(all_lw.keys()[ii]), 'f8', ('level', ))
            deriv_ri_lw_out[ii] = outfile.createVariable("difference_quotient_ri_lw_{}".format(all_lw.keys()[ii]), 'f8', ('level', ))
            deriv_wpi_lw_out[ii] = outfile.createVariable("difference_quotient_wpi_lw_{}".format(all_lw.keys()[ii]), 'f8', ('level', ))

            if all_lw.keys()[ii] == "HEATING RATE":
                all_lw_out[ii].units = "degree/day"
                clear_lw_out[ii].units = "degree/day"
                deriv_cwp_lw_out[ii].units = "degree/day"
                deriv_rl_lw_out[ii].units = "degree/day"
                deriv_ri_lw_out[ii].units = "degree/day"
                deriv_wpi_lw_out[ii].units = "degree/day"

            else:
                all_lw_out[ii].units = "Wm-2"
                clear_lw_out[ii].units = "Wm-2"
                deriv_cwp_lw_out[ii].units = "Wm-2"
                deriv_rl_lw_out[ii].units = "Wm-2"
                deriv_ri_lw_out[ii].units = "Wm-2"
                deriv_wpi_lw_out[ii].units = "Wm-2"

            all_lw_out[ii][:] = all_lw[all_lw.keys()[ii]][::-1]
            clear_lw_out[ii][:] = clear_lw[clear_lw.keys()[ii]][::-1]
            deriv_cwp_lw_out[ii][:] = deriv_cwp_lw[all_lw.keys()[ii]][::-1]
            deriv_rl_lw_out[ii][:] = deriv_rl_lw[all_lw.keys()[ii]][::-1]
            deriv_ri_lw_out[ii][:] = deriv_ri_lw[all_lw.keys()[ii]][::-1]
            deriv_wpi_lw_out[ii][:] = deriv_wpi_lw[all_lw.keys()[ii]][::-1]
        
def main(lat, lon, sza, cwp, wpi, rl, ri, cloud, z, t, q, p, dcwp, dwpi, drl, dri, co2, n2o, ch4, albedo_dir, albedo_diff, iceconc, semiss, clt, fname):
    
    if sza > 90.0:
        sza = 90.0
        
    albedo_dir = np.float(albedo_dir)
    albedo_diff = np.float(albedo_diff)
    iceconc = np.float(iceconc)
    
    clear_sw, clear_lw = calc_clear_sky(z, p, t, q, sza, albedo_dir, albedo_diff, semiss, lat, co2, n2o, ch4)

    delta = 0.01
    delta_wpi = 0.01
    if np.mean(wpi) == 1:
        delta_wpi = -0.01
    all_sw, all_lw = calc_all_sky(z, p, t, q, sza, albedo_dir, albedo_diff, cloud, cwp, rl, ri, wpi, semiss, lat, co2, n2o, ch4, clt)
    dcwp_sw, dcwp_lw = calc_all_sky(z, p, t, q, sza, albedo_dir, albedo_diff, cloud, cwp+delta, rl, ri, wpi, semiss, lat, co2, n2o, ch4, clt)
    drl_sw, drl_lw = calc_all_sky(z, p, t, q, sza, albedo_dir, albedo_diff, cloud, cwp, rl+delta, ri, wpi, semiss, lat, co2, n2o, ch4, clt)
    dri_sw, dri_lw = calc_all_sky(z, p, t, q, sza, albedo_dir, albedo_diff, cloud, cwp, rl, ri+delta, wpi, semiss, lat,  co2, n2o, ch4, clt)   
    dwpi_sw, dwpi_lw = calc_all_sky(z, p, t, q, sza, albedo_dir, albedo_diff, cloud, cwp, rl, ri, wpi+delta_wpi, semiss, lat, co2, n2o, ch4, clt)   
    
    deriv_cwp_lw = error_propagation(dcwp_lw, all_lw, delta)
    deriv_cwp_sw = error_propagation(dcwp_sw, all_sw, delta)
    deriv_rl_lw = error_propagation(drl_lw, all_lw, delta)
    deriv_rl_sw = error_propagation(drl_sw, all_sw, delta)
    deriv_ri_lw = error_propagation(dri_lw, all_lw, delta)
    deriv_ri_sw = error_propagation(dri_sw, all_sw, delta)
    deriv_wpi_lw = error_propagation(dwpi_lw, all_lw, delta_wpi)
    deriv_wpi_sw = error_propagation(dwpi_sw, all_sw, delta_wpi)
          
    write_results(all_sw, all_lw, clear_sw, clear_lw, deriv_cwp_lw, deriv_cwp_sw, deriv_rl_lw, deriv_rl_sw, deriv_ri_lw, deriv_ri_sw, deriv_wpi_lw, deriv_wpi_sw, lat, lon, sza, cwp, wpi, rl, ri, cloud, z, t, q, p, dcwp, dwpi, drl, dri, co2, n2o, ch4, albedo_dir, albedo_diff, iceconc, semiss, clt, fname)

    os.remove('TAPE6')
    os.remove('TAPE7')
    os.remove('tape6')
    os.remove('INPUT_RRTM')
    os.remove('IN_CLD_RRTM')
    os.remove('OUTPUT_RRTM')

    return 1