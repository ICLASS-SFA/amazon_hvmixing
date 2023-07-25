"""
Calculate W vertical profile statistics within each MCS from pre-processed WRF data on HAMSL and 
saves the output matching MCS tracks to a netCDF file.
This version processes a subset of tracks provided from input arguments.
"""
__author__ = "Zhe.Feng@pnnl.gov"
__date__ = "28-Jun-2023"

import numpy as np
import xarray as xr
import pandas as pd
import sys, os
import time
import yaml
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, binary_dilation, iterate_structure
import warnings
import dask
from dask.distributed import Client, LocalCluster

#-----------------------------------------------------------------------
def label_cores(W, W_thresh, ncores_min, min_core_npix, method='>'):
    """
    Label up/down draft cores using threshold and connectivity.

    Args:
        W: np.array
            Vertical velocity array
        W_thresh: float
            Vertical velocity threshold
        ncores_min: int
            Minimum number of cores to save
        min_core_npix: int
            Minimum number of pixels to define a core
        method: string
            Method to define cores

    Returns:
        ncores_save: int
            Number of cores
        npix_core_sorted: np.array
            Number of pixels for each core
        core_numbers_sorted: np.array
            Labeled core numbers (1D)
        core_label: np.array
            Labeled core numbers map (2D)
    """
    # Label cores at a given vertical level
    if method == '>':
        core_label, ncores = ndimage.label((W > W_thresh))
    elif method == '<':
        core_label, ncores = ndimage.label((W < W_thresh))
    else:
        print(f'Error: Undefined method to label cores: {method}!')
    
    # Get core sizes
    core_numbers, npix_core = np.unique(core_label, return_counts=True)
    
    # Remove 0 label result since that is the background
    npix_core = npix_core[core_numbers > 0]
    core_numbers = core_numbers[core_numbers > 0]
    
    # Remove small cores
    mask = npix_core > min_core_npix
    npix_core = npix_core[mask]
    core_numbers = core_numbers[mask]

    # Sort the core size by descending order
    sort_idx = npix_core.argsort()[::-1]
    npix_core_sorted = npix_core[sort_idx]
    core_numbers_sorted = core_numbers[sort_idx]
    
    # Save the largest X cores
    ncores_all = len(npix_core)
    ncores_save = np.nanmin([ncores_all, ncores_min])
    
    # Put output variables in a dictionary
    out_dict = {
        'ncores_all': ncores_all,
        'ncores_save': ncores_save,
        'core_npix': npix_core_sorted[:ncores_save+1],
        'core_numbers': core_numbers_sorted[:ncores_save+1],
        'core_label': core_label,
    }
    return out_dict

#-----------------------------------------------------------------------
def calc_w_prof(
    filename_mcsmask, 
    filename_data,      
    idx_track,
    config,
):
    """
    Calculate W statistics for MCS in a single pixel file

    Args:
        filename_mcsmask: string
            MCS tracking pixel filename
        filename_data: string
            W data filename
        idx_track: np.array
            Tracknumber indices in the pixel file
        config: dictionary
            Dictionary containing config parameters

    Returns:
        out_dict: dictionary
            Dictionary containing the track statistics data
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data
    """

    # Get thresholds from config
    W_up_thresh = config['W_up_thresh']
    W_down_thresh = config['W_down_thresh']
    Q_up_thresh = config['Q_up_thresh']
    min_core_npix = config['min_core_npix']
    ncores_min = config['ncores_min']

    # Ideal gas constants
    R_dry = 287.058   # [J kg−1 K−1] gas constant for dry air
    R_v = 461         # [J K-1 kg-1] gas constant for water vapor
    Cp_dry = 1005.7   # [J kg-1 K-1] specific heat capacity for dry air 
    Epsilon = R_dry / R_v

    # Read MCS mask
    dsm = xr.open_dataset(filename_mcsmask)

    # Read 3D file
    dsw = xr.open_dataset(filename_data)

    # Rename coordinates and dimensions in MCS mask DataSet
    dsm = dsm.rename_vars({'lat':'lat2d', 'lon':'lon2d'})
    dsm = dsm.rename_dims({'y':'lat', 'x':'lon'})
    # Assign 1D coordinates to both DataSets
    dsm = dsm.assign_coords({'lon':dsw['lon'], 'lat':dsw['lat']})
    dsw = dsw.assign_coords({'lon':dsw['lon'], 'lat':dsw['lat']})

    # Get MCS mask variables
    mcs_tracknumbers = dsm['cloudtracknumber'].squeeze()
    # Get 3D variables
    nz = dsw.sizes['height']

    # MCS track numbers in mask file is shifted by 1
    track_numbers = idx_track + 1

    out_dict = None
    out_dict_attrs = None

    # Proceed if number of matched MCS is > 0
    nmatchmcs = len(idx_track)
    if (nmatchmcs > 0):

        # Get WRF variables
        DX = dsw.attrs['DX']
        DY = dsw.attrs['DY']
        grid_area = DX * DY / 1e6   # [km^2]

        PRESSURE = dsw['P'].squeeze()
        TH = dsw['THETA'].squeeze()
        TV = dsw['TV'].squeeze()
        THETA_E = dsw['THETA_E'].squeeze()
        QV = dsw['QVAPOR'].squeeze()
        # QCLOUD = dsw['QCLOUD'].squeeze()
        # QT = dsw['QTOTAL'].squeeze()
        WA = dsw['W'].squeeze()

        # Make array to store output
        dims2d = (nmatchmcs, nz)
        dims3d = (nmatchmcs, nz, ncores_min)
        nCore_up = np.full(dims2d, np.NaN, dtype=np.float32)
        MassFlux_up = np.full(dims2d, np.NaN, dtype=np.float32)
        CoreArea_up = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreMaxW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreMeanW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreThtvMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreThtvMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreBuoyThtv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreThteMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreThteMean_up = np.full(dims3d, np.NaN, dtype=np.float32)

        nCore_down = np.full(dims2d, np.NaN, dtype=np.float32)
        MassFlux_down = np.full(dims2d, np.NaN, dtype=np.float32)
        CoreArea_down = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreMinW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        CoreMeanW_down = np.full(dims3d, np.NaN, dtype=np.float32)

        # Core dilation structure to make perimeter
        struct = generate_binary_structure(2, 2)

        # Loop over each MCS
        for itrack, track_num in enumerate(track_numbers):
            # print(itrack, track_num)
            print(f'MCS track {track_num} ...')

            # Get MCS mask and apply to the 3D variable
            mcsmask = mcs_tracknumbers == track_num

            # Count the number of pixels for the MCS mask
            inpix_cloud = np.count_nonzero(mcsmask)
            if inpix_cloud > 0:

                # Calculate statistics
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # warnings.simplefilter("ignore", category=UserWarning)

                    # Subset 3D variables to the current mask
                    iPRESSURE = PRESSURE.where(mcsmask == True, drop=True).squeeze().data

                    # Proceed if subsetted variable dimension is 3D
                    if iPRESSURE.ndim == 3:

                        iW = WA.where(mcsmask == True, drop=True).squeeze().data
                        iTH = TH.where(mcsmask == True, drop=True).squeeze().data
                        iTV = TV.where(mcsmask == True, drop=True).squeeze().data
                        iThte = THETA_E.where(mcsmask == True, drop=True).squeeze().data
                        iQV = QV.where(mcsmask == True, drop=True).squeeze().data
                        # iQCLOUD = QCLOUD.where(mcsmask == True, drop=True).squeeze().data
                        # iQT = QT.where(mcsmask == True, drop=True).squeeze().data
                        
                        # Moist air density [kg m-3]
                        iRho = iPRESSURE / (R_dry * iTV)
                        # Mass flux (kg m-2 s-1)
                        iMassFlux = (iRho * iW).squeeze()
                        
                        # Calculate Virtual Potential Temperature
                        iThtv = iTH * (iQV + 0.622)/(0.622 * (1 + iQV))

                        # # Vapor pressure [Pa]
                        # VaporP = iPRESSURE * iQV / (Epsilon + iQV)
                        # # Calculate air temperature (AMS)
                        # iTK = iTH * (iPRESSURE / 1000)**(2/7) # Remember that PRESSURE is in hPa
                        # # Dry air temperature [K]
                        # iTK = iTH * (1e5 / (iPRESSURE - VaporP)) ** (- R_dry / Cp_dry)

                        # # Calculate Density Temperature (Eqn. 4.3.6 of some Emanuel textbook)
                        # # "Note that Tv is a special case of Trho, since when condensed water is absent rT = r."
                        # iTrho = iTK * (1 + iQV / 0.622) / (1 + iQT)
                        # import pdb; pdb.set_trace()

                        # Loop over vertical level
                        for z in range(0, nz):
                            # print(height[z])
                            zW = iW[z,:,:]
                            # zQCLOUD = iQCLOUD[z,:,:]
                            zMassFlux = iMassFlux[z,:,:]
                            zThtv = iThtv[z,:,:]
                            zThte = iThte[z,:,:]

                            # Proceed if max(W) > threshold
                            if np.nanmax(zW) > W_up_thresh:

                                # # Cloudy updrafts (should incloud QICE too)
                                # zWcloud = (zW > W_up_thresh) & (zQCLOUD > Q_up_thresh)
                                # # Make cloudy updraft mask
                                # zW_mask = np.zeros_like(zW)
                                # zW_mask[zWcloud == True] = 1

                                # Label updraft cores
                                # dict_up = label_cores(zW * zW_mask, W_up_thresh, ncores_min, min_core_npix, method='>')
                                dict_up = label_cores(zW, W_up_thresh, ncores_min, min_core_npix, method='>')
                                ncores_all_up = dict_up['ncores_all']
                                ncores_up = dict_up['ncores_save']
                                core_npix_up = dict_up['core_npix']
                                core_numbers_up = dict_up['core_numbers']
                                core_label_up = dict_up['core_label']

                                # Get the min number of cores to save
                                ncores_save_up = min([ncores_up, ncores_min])

                                # Make core perimeter masks
                                # core_label_up_dil = np.zeros_like(core_label_up)
                                core_label_up_prm = np.zeros_like(core_label_up)
                                for ii in range(ncores_save_up):
                                    cell = np.zeros_like(core_label_up)
                                    cell[core_label_up == core_numbers_up[ii]] = 1
                                    dil = binary_dilation(cell, structure=struct, iterations=1)
                                    # core_label_up_dil[dil == 1] = core_numbers_up[ii] 
                                    core_label_up_prm[(dil - cell) == 1] = core_numbers_up[ii]

                                # if (ncores_save_up > 1):
                                #     if np.nanmax(core_npix_up) > 20:
                                #         import matplotlib.pyplot as plt
                                #         import pdb; pdb.set_trace()

                                # Total number of cores
                                nCore_up[itrack, z] = ncores_all_up

                                # Calculate core statistics
                                W_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                                W_mean_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                                Thtv_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                                Thtv_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                                Buoy_Thtv_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                                Thte_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                                Thte_mean_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)

                                for ii in range(ncores_save_up):
                                    # MaFlx_sum_up[ii] = np.nansum(zMassFlux[core_label_up == core_numbers_up[ii]])
                                    W_max_up[ii] = np.nanmax(zW[core_label_up == core_numbers_up[ii]])
                                    W_mean_up[ii] = np.nanmean(zW[core_label_up == core_numbers_up[ii]])
                                    Thtv_max_up[ii] = np.nanmax(zThtv[core_label_up == core_numbers_up[ii]])
                                    Thtv_mean_prm[ii] = np.nanmean(zThtv[core_label_up_prm == core_numbers_up[ii]])
                                    Buoy_Thtv_up[ii] = 9.81*(Thtv_max_up[ii]-Thtv_mean_prm[ii])/Thtv_mean_prm[ii]
                                    Thte_max_up[ii] = np.nanmax(zThte[core_label_up == core_numbers_up[ii]])
                                    Thte_mean_up[ii] = np.nanmean(zThte[core_label_up == core_numbers_up[ii]])
                                # Calculate total mass flux for all labeled cores
                                MaFlx_sum_up = np.nansum(zMassFlux[core_label_up > 0])

                                # Save data to output arrays
                                # ncores_save_up = min([ncores_up, ncores_min])
                                MassFlux_up[itrack, z] = MaFlx_sum_up * DX * DY
                                # MassFlux_up[itrack, z, 0:ncores_save_up] = MaFlx_sum_up[0:ncores_save_up] * DX * DY
                                CoreArea_up[itrack, z, 0:ncores_save_up] = core_npix_up[0:ncores_save_up] * grid_area
                                CoreMaxW_up[itrack, z, 0:ncores_save_up] = W_max_up[0:ncores_save_up]
                                CoreMeanW_up[itrack, z, 0:ncores_save_up] = W_mean_up[0:ncores_save_up]
                                CoreThtvMax_up[itrack, z, 0:ncores_save_up] = Thtv_max_up[0:ncores_save_up]
                                CoreThtvMean_prm[itrack, z, 0:ncores_save_up] = Thtv_mean_prm[0:ncores_save_up]
                                CoreBuoyThtv_up[itrack, z, 0:ncores_save_up] = Buoy_Thtv_up[0:ncores_save_up]
                                CoreThteMax_up[itrack, z, 0:ncores_save_up] = Thte_max_up[0:ncores_save_up]
                                CoreThteMean_up[itrack, z, 0:ncores_save_up] = Thte_mean_up[0:ncores_save_up]


                            # Proceed if min(W) < threshold
                            if np.nanmin(zW) < W_down_thresh:
                                # Label downdraft cores
                                dict_down = label_cores(zW, W_down_thresh, ncores_min, min_core_npix, method='<')
                                ncores_all_down = dict_down['ncores_all']
                                ncores_down = dict_down['ncores_save']
                                core_npix_down = dict_down['core_npix']
                                core_numbers_down = dict_down['core_numbers']
                                core_label_down = dict_down['core_label']

                                # Total number of cores
                                nCore_down[itrack, z] = ncores_all_down

                                # Calculate core statistics
                                W_min_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                                W_mean_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                                # Thtv_min_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                                # Thtv_mean_prm = np.full(ncores_down, np.NaN, dtype=np.float32)
                                # Buoy_Thtv_down = np.full(ncores_down, np.NaN, dtype=np.float32)

                                for ii in range(ncores_down):
                                    W_min_down[ii] = np.nanmin(zW[core_label_down == core_numbers_down[ii]])
                                    W_mean_down[ii] = np.nanmean(zW[core_label_down == core_numbers_down[ii]])
                                    # Thtv_min_down[ii] = np.nanmin(zThtv[core_label_down == core_numbers_down[ii]])
                                    # Thtv_mean_prm[ii] = np.nanmean(zThtv[core_label_down_prm == core_numbers_down[ii]])
                                    # Buoy_Thtv_down[ii] = 9.81*(Thtv_min_down[ii]-Thtv_mean_prm[ii])/Thtv_mean_prm[ii]
                                # Calculate total mass flux for all labeled cores
                                MaFlx_sum_down = np.nansum(zMassFlux[core_label_down > 0])

                                # Save data to output arrays
                                ncores_save_down = min([ncores_down, ncores_min])
                                MassFlux_down[itrack, z] = MaFlx_sum_down * DX * DY
                                CoreArea_down[itrack, z, 0:ncores_save_down] = core_npix_down[0:ncores_save_down] * grid_area
                                CoreMinW_down[itrack, z, 0:ncores_save_down] = W_min_down[0:ncores_save_down]
                                CoreMeanW_down[itrack, z, 0:ncores_save_down] = W_mean_down[0:ncores_save_down]

        # Group outputs in dictionaries
        out_dict = {
            'nCore_up': nCore_up,
            'CoreArea_up': CoreArea_up,
            'CoreMaxW_up': CoreMaxW_up,
            'CoreMeanW_up': CoreMeanW_up,
            'MassFlux_up': MassFlux_up,
            'CoreThteMax_up': CoreThteMax_up,
            'CoreThteMean_up': CoreThteMean_up,
            'CoreThtvMax_up': CoreThtvMax_up,
            'CoreThtvMean_prm': CoreThtvMean_prm,
            'CoreBuoyThtv_up': CoreBuoyThtv_up,

            'nCore_down': nCore_down,
            'CoreArea_down': CoreArea_down,
            'CoreMinW_down': CoreMinW_down,
            'CoreMeanW_down': CoreMeanW_down,
            'MassFlux_down': MassFlux_down,
        }
        out_dict_attrs = {
            # Updraft
            'nCore_up': {
                'long_name': 'Number of updraft cores',
                'units': 'count',
            },
            'MassFlux_up': {
                'long_name': 'Total updraft mass flux',
                'units': 'kg s^-1',
            },
            'CoreArea_up': {
                'long_name': 'Updraft core area',
                'units': 'km^2',
            },
            'CoreMaxW_up': {
                'long_name': 'Updraft core maximum W',
                'units': 'm/s',
            },
            'CoreMeanW_up': {
                'long_name': 'Updraft core mean W',
                'units': 'm/s',
            },
            'CoreThteMax_up': {
                'long_name': 'Updraft core max Theta e',
                'units': 'K',
            },
            'CoreThteMean_up': {
                'long_name': 'Updraft core mean Theta e',
                'units': 'K',
            },
            'CoreThtvMax_up': {
                'long_name': 'Updraft core max Theta v',
                'units': 'K',
            },
            'CoreThtvMean_prm': {
                'long_name': 'Updraft core perimeter mean Theta v',
                'units': 'K',
            },
            'CoreBuoyThtv_up': {
                'long_name': 'Updraft core Buoyancy based on Theta v',
                'units': 'm s^-2',
            },
            # Downdraft
            'nCore_down': {
                'long_name': 'Number of downdraft cores',
                'units': 'count',
            },
            'CoreArea_down': {
                'long_name': 'Downdraft core area',
                'units': 'km^2',
            },
            'CoreMinW_down': {
                'long_name': 'Downdraft core minimum W',
                'units': 'm/s',
            },
            'CoreMeanW_down': {
                'long_name': 'Downdraft core mean W',
                'units': 'm/s',
            },
            'MassFlux_down': {
                'long_name': 'Total downdraft mass flux',
                'units': 'kg s^-1',
            },
        }
        print(f'Done processing: {filename_data}')

    return out_dict, out_dict_attrs


def main():

    # Get the number of command line arguments
    nargs = len(sys.argv)
    if nargs == 5:
        config_file = sys.argv[1]
        track_start = int(sys.argv[2])
        track_end = int(sys.argv[3])
        digits = int(sys.argv[4])
    elif nargs == 6:
        config_file = sys.argv[1]
        track_start = int(sys.argv[2])
        track_end = int(sys.argv[3])
        digits = int(sys.argv[4])
        scheduler_file = sys.argv[5]

    # Get inputs from configuration file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    track_period = config['track_period']
    data_dir = config['data_dir']
    mcsmask_dir = config['mcsmask_dir'] + track_period + '/'
    mcsstats_dir = config['mcsstats_dir']
    output_dir = config['output_dir']
    statsfile_basename = config['statsfile_basename']
    mcsmask_basename = config['mcsmask_basename']
    data_basename = config['data_basename']
    outputfile_basename = config['outputfile_basename']
    ncores_min = config['ncores_min']
    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']

    # Track statistics file
    trackstats_file = f"{mcsstats_dir}{statsfile_basename}{track_period}.nc"
    # Output statistics filename
    # output_filename = f"{output_dir}{outputfile_basename}W_{track_period}.nc"
    os.makedirs(output_dir, exist_ok=True)

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'
    z_dimname = 'level'
    core_dimname = 'core'

    # Read robust MCS statistics
    dsstats = xr.open_dataset(trackstats_file)

    # Make output filename
    track_start_str = str(track_start).zfill(digits)
    output_filename = f"{output_dir}{outputfile_basename}W_{track_period}_t{track_start_str}.nc"

    # Select tracks for this part        
    coord_tracks = dsstats['tracks'].isel(tracks=slice(track_start, track_end))
    coord_times = dsstats['times']
    rmcs_basetime = dsstats['base_time'].isel(tracks=slice(track_start, track_end))
    ntracks = len(coord_tracks)

    # ntracks = dsstats.sizes['tracks']
    ntimes = dsstats.sizes['times']
    # stats_basetime = dsstats['base_time']

    # Get end times for all tracks
    # Sum over time dimension for valid basetime indices, -1 to get the last valid time index for each track
    # This is the end time index of each track (i.e. +1 equals the lifetime of each track)
    end_time_idx = np.sum(np.isfinite(rmcs_basetime), axis=1)-1
    # Apply fancy indexing to base_time: a tuple that indicates for each track, get the end time index
    end_basetime = rmcs_basetime[(np.arange(0,ntracks), end_time_idx)]

    # Find all unique valid MCS times from the track data
    unique_rmcs_times = np.unique(rmcs_basetime.values[~np.isnan(rmcs_basetime)])
    nfiles = len(unique_rmcs_times)
    print(f"Total number of MCS files: {nfiles}")

    # Create a list to store matchindices for each ERA5 file
    trackindices_all = []
    timeindices_all = []
    results = []

    if run_parallel == 1:
        # Initialize dask
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
    if run_parallel == 2:
        # Dask-MPI
        # scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler.json")
        client = Client(scheduler_file=scheduler_file)

    # Loop over each MCS mask file
    for ifile in range(nfiles):
        idatetime = unique_rmcs_times[ifile]
        dt_str1 = pd.to_datetime(idatetime).strftime("%Y%m%d_%H%M%S")
        dt_str2 = pd.to_datetime(idatetime).strftime("%Y-%m-%d_%H:%M:%S")
        filename_mcsmask = f"{mcsmask_dir}{mcsmask_basename}{dt_str1}.nc"
        filename_data = f"{data_dir}{data_basename}{dt_str2}.nc"
        # print(filename_mcsmask)
        print(filename_data)

        # Get all MCS tracks/times indices that are in the same time
        idx_track, idx_time = np.where(rmcs_basetime == idatetime)
        # Get track coordinates (indices w.r.t. all tracks)
        icoord_track = coord_tracks.data[idx_track]
        # import pdb; pdb.set_trace()

        if len(icoord_track) > 0:
            # Save track/time indices for the current file to the overall list
            trackindices_all.append(idx_track)
            timeindices_all.append(idx_time)
            # Serial
            if run_parallel == 0:
                result = calc_w_prof(
                    filename_mcsmask, 
                    filename_data,                  
                    icoord_track, 
                    config,
                )
            # Parallel
            elif run_parallel >= 1:
                result = dask.delayed(calc_w_prof)(
                    filename_mcsmask, 
                    filename_data,                  
                    icoord_track, 
                    config,
            )
            results.append(result)
    
    if run_parallel >= 1:
        # Trigger Dask computation
        print("Computing statistics ...")
        final_results = dask.compute(*results)

    
    # Read a 3D file to get vertical coordinates
    dsw = xr.open_dataset(filename_data)
    nz = dsw.dims['height']
    height = dsw['height']
    DX = dsw.attrs['DX']
    DY = dsw.attrs['DY']
    dsw.close()

    # Make a variable list and get attributes from one of the returned dictionaries
    # Loop over each return results till one that is not None
    counter = 0
    while counter < nfiles:
        if final_results[counter] is not None:
            var_names = list(final_results[counter][0].keys())
            # Get variable attributes
            var_attrs = final_results[counter][1]
            break
        counter += 1

    # Loop over variable list to create the dictionary entry
    out_dict = {}
    out_dict_attrs = {}
    for ivar in var_names:
        if ('nCore_' in ivar) or ('MassFlux_' in ivar):
            out_dict[ivar] = np.full((ntracks, ntimes, nz), np.nan, dtype=np.float32)
        else:
            out_dict[ivar] = np.full((ntracks, ntimes, nz, ncores_min), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]

    # Now that all calculations for each pixel file is done, put the results back to the tracks format
    # Loop over the returned statistics list
    for ifile in range(len(final_results)):
        # Get the results from the current file
        vars = final_results[ifile]
        if (vars is not None):
            # Get the return results for this pixel file
            # The result is a tuple: (out_dict, out_dict_attrs)
            # The first entry is the dictionary containing the variables
            iVAR = final_results[ifile][0]

            # Get trackindices and timeindices for this file
            trackindices = trackindices_all[ifile]
            timeindices = timeindices_all[ifile]
            # import pdb; pdb.set_trace()

            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                if iVAR[ivar].ndim == 2:
                    out_dict[ivar][trackindices,timeindices,:] = iVAR[ivar]
                if iVAR[ivar].ndim == 3:
                    out_dict[ivar][trackindices,timeindices,:,:] = iVAR[ivar]

    ##########################################################
    # Write to netcdf
    print('Writing output netcdf ... ')

    # Define variable list
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if value.ndim == 2:
            var_dict[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
        if value.ndim == 3:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname], value, out_dict_attrs[key])
        if value.ndim == 4:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname, core_dimname], value, out_dict_attrs[key])
    # Add base_time from track stats to the output dictionary
    out_dict['base_time'] = ([tracks_dimname, times_dimname], rmcs_basetime.data, rmcs_basetime.attrs)
    # Define coordinate list
    core_dim_attrs = {
        'long_name': 'Core number',
    }
    coord_dict = {
        # tracks_dimname: ([tracks_dimname], np.arange(0, ntracks)),
        tracks_dimname: ([tracks_dimname], coord_tracks.data, coord_tracks.attrs),
        times_dimname: ([times_dimname], coord_times.data, coord_times.attrs),
        z_dimname: ([z_dimname], height.data, height.attrs),
        core_dimname: ([core_dimname], np.arange(0, ncores_min), core_dim_attrs),
    }
    # Define global attributes
    gattr_dict = {
        'title':  'MCS W statistics', \
        'Institution': 'Pacific Northwest National Laboratoy', \
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
        'Created_on':  time.ctime(time.time()), \
        'source_trackfile': trackstats_file, \
        'DX': DX,
        'DY': DY,
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Delete file if it already exists
    if os.path.isfile(output_filename):
        os.remove(output_filename)
        
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {output_filename}')


if __name__ == "__main__":
    main()

    