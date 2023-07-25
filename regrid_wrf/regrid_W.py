from __future__ import print_function
import os, sys
import datetime, time, calendar
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from wrf import getvar, vinterp, extract_times

if __name__ == '__main__':
    # Get runname, filename from input
    filein = str(sys.argv[1])
    outdir = str(sys.argv[2])

    # Set file directories
    outbasename = 'wrfreg_'

    # Define regrid vertical levels
    # lev = np.arange(0.5, 18.0, 0.5)
    # lev = np.arange(0.25, 18.0, 0.25)
    lev_low = np.arange(0.25, 6.01, 0.25)
    lev_hi = np.arange(6.5, 18.01, 0.5)
    lev = np.concatenate((lev_low, lev_hi))
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # filename
    #filein = indir + fname
    print('Reading input: ', filein)
    # Get date/time string from filename
    # assuming wrfout file has this standard format: wrfout_d0x_yyyy-mm-dd_hh:mm:ss
    fname = os.path.basename(filein)
    ftime = fname[11:]
    fileout = f'{outdir}/{outbasename}{ftime}.nc'

    # Read WRF file
    ncfile = Dataset(filein)
    # Read Times as characters directly
    times_char = ncfile.variables['Times'][:]
    # Use WRF routine to read time as np.datetime64
    rawtime = extract_times(ncfile, None)
    rawtime = np.datetime64(rawtime[0])

    # Convert np.datetime64 to datetime object
    dt = datetime.datetime.utcfromtimestamp(rawtime.astype('O')/1e9)
    # Convert datetime to base_time
    base_time = calendar.timegm(dt.timetuple())
    # Create a numpy array for Xarray time dimension
    btarr = np.array([base_time], np.int64)

    # Convert datetime object to WRF string format
    Times_str = dt.strftime('%Y-%m-%d_%H:%M:%S')
    print(Times_str)
    strlen = len(Times_str)

    DX = getattr(ncfile, 'DX')
    DY = getattr(ncfile, 'DY')

    # Read variables
    XLAT = getvar(ncfile, "XLAT")
    XLONG = getvar(ncfile, "XLONG")

    # Get dimension
    ny, nx = np.shape(XLAT)

    # Interpolate to specified height levels
    # dbz = getvar(ncfile, 'REFL_10CM')
    # # Convert reflectivity to linear unit
    # refl = 10.0**(dbz/10.0)
    # refl_reg = vinterp(ncfile, field=refl, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    # # Convert linear reflectivity to dBZ unit
    # dbz_reg = 10.0 * np.log10(refl_reg)

    # Microphysics variables
    QVAPOR = getvar(ncfile, 'QVAPOR')
    QVAPOR_reg = vinterp(ncfile, field=QVAPOR, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)

    QRAIN = getvar(ncfile, 'QRAIN')
    # QRAIN_reg = vinterp(ncfile, field=QRAIN, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    QGRAUP = getvar(ncfile, 'QGRAUP')
    # QGRAUP_reg = vinterp(ncfile, field=QGRAUP, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    QSNOW = getvar(ncfile, 'QSNOW')
    # QSNOW_reg = vinterp(ncfile, field=QSNOW, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    QCLOUD = getvar(ncfile, 'QCLOUD')
    QCLOUD_reg = vinterp(ncfile, field=QCLOUD, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    QICE = getvar(ncfile, 'QICE')
    # QICE_reg = vinterp(ncfile, field=QICE, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    # QNRAIN = getvar(ncfile, 'QNRAIN')
    # QNRAIN_reg = vinterp(ncfile, field=QNRAIN, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    # QNICE = getvar(ncfile, 'QNICE')
    # QNICE_reg = vinterp(ncfile, field=QNICE, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    QTOTAL = QVAPOR + QRAIN + QGRAUP + QSNOW + QCLOUD + QICE
    QTOTAL_reg = vinterp(ncfile, field=QTOTAL, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)
    # import pdb; pdb.set_trace()

    # Interpolate variables
    wa = getvar(ncfile, 'wa')
    W_reg = vinterp(ncfile, field=wa, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)

    # heating = getvar(ncfile, 'H_DIABATIC')
    # # Convert reflectivity to linear unit
    # heating_reg = vinterp(ncfile, field=heating, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)

    th = getvar(ncfile, 'th')
    th_reg = vinterp(ncfile, field=th, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)

    tv = getvar(ncfile, 'tv')
    tv_reg = vinterp(ncfile, field=tv, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)

    pressure = getvar(ncfile, 'pressure')
    p_reg = vinterp(ncfile, field=pressure, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)

    theta_e = getvar(ncfile, 'theta_e')
    theta_e_reg = vinterp(ncfile, field=theta_e, vert_coord='ght_msl', interp_levels=lev, extrapolate=False)


    # Remove attributes ('projection' in particular conflicts with Xarray)
    attrs_to_remove = ['FieldType', 'projection', 'MemoryOrder', 'stagger', 'coordinates', 'missing_value']
    for key in attrs_to_remove:
        XLONG.attrs.pop(key, None)
        XLAT.attrs.pop(key, None)
        W_reg.attrs.pop(key, None)
        # heating_reg.attrs.pop(key, None)
        # QRAIN_reg.attrs.pop(key, None)
        # QGRAUP_reg.attrs.pop(key, None)
        # QSNOW_reg.attrs.pop(key, None)
        QCLOUD_reg.attrs.pop(key, None)
        # QICE_reg.attrs.pop(key, None)
        # QNRAIN_reg.attrs.pop(key, None)
        # QNICE_reg.attrs.pop(key, None)
        QVAPOR_reg.attrs.pop(key, None)
        QTOTAL_reg.attrs.pop(key, None)
        th_reg.attrs.pop(key, None)
        tv_reg.attrs.pop(key, None)
        p_reg.attrs.pop(key, None)
        theta_e_reg.attrs.pop(key, None)

    dim4d = ['time', 'height', 'lat', 'lon']
    var_dict = {
        'base_time': (['time'], btarr), \
        'lon2d': (['lat', 'lon'], XLONG.data, XLONG.attrs),
        'lat2d': (['lat', 'lon'], XLAT.data, XLAT.attrs),
        'W': (dim4d, W_reg.expand_dims('time', axis=0).data, W_reg.attrs),
        # 'LH': (dim4d, heating_reg.expand_dims('time', axis=0).data, heating_reg.attrs),
        # 'QRAIN': (dim4d, QRAIN_reg.expand_dims('time', axis=0).data, QRAIN_reg.attrs),
        # 'QGRAUP': (dim4d, QGRAUP_reg.expand_dims('time', axis=0).data, QGRAUP_reg.attrs),
        # 'QSNOW': (dim4d, QSNOW_reg.expand_dims('time', axis=0).data, QSNOW_reg.attrs),
        'QCLOUD': (dim4d, QCLOUD_reg.expand_dims('time', axis=0).data, QCLOUD_reg.attrs),
        # 'QICE': (dim4d, QICE_reg.expand_dims('time', axis=0).data, QICE_reg.attrs),
        # 'QNRAIN': (dim4d, QNRAIN_reg.expand_dims('time', axis=0).data, QNRAIN_reg.attrs),
        # 'QNICE': (dim4d, QNICE_reg.expand_dims('time', axis=0).data, QNICE_reg.attrs),
        'QVAPOR': (dim4d, QVAPOR_reg.expand_dims('time', axis=0).data, QVAPOR_reg.attrs),
        'QTOTAL': (dim4d, QTOTAL_reg.expand_dims('time', axis=0).data, QTOTAL_reg.attrs),
        'THETA': (dim4d, th_reg.expand_dims('time', axis=0).data, th_reg.attrs),
        'TV': (dim4d, tv_reg.expand_dims('time', axis=0).data, tv_reg.attrs),
        'P': (dim4d, p_reg.expand_dims('time', axis=0).data, p_reg.attrs),
        'THETA_E': (dim4d, theta_e_reg.expand_dims('time', axis=0).data, theta_e_reg.attrs),
    }
    coord_dict = {
        'time': (['time'], btarr),
        'height': (['height'], lev),
    }
    gattr_dict = {
        'Title': 'WRF regridded post processed data',
        'contact': 'Zhe Feng: zhe.feng@pnnl.gov',
        'created_on': time.ctime(time.time()),
        'Institution': 'Pacific Northwest National Laboratory',
        'Original_File': filein,
        'DX': DX,
        'DY': DY,
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Specify attributes
    dsout['base_time'].attrs['long_name'] = 'Epoch time (seconds since 1970-01-01 00:00:00)'
    dsout['base_time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    dsout['time'].attrs['long_name'] = 'Epoch time (seconds since 1970-01-01 00:00:00)'
    dsout['time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    dsout['height'].attrs['long_name'] = 'Height above mean sea level'
    dsout['height'].attrs['units'] = 'km'
    dsout['QTOTAL'].attrs['long_name'] = 'Total water mixing ratio'
    dsout['QTOTAL'].attrs['units'] = 'kg kg-1'
    # dsout.Times.attrs['long_name'] = 'WRF-based time'
    # dsout.lon2d.attrs['long_name'] = 'Longitude'
    # dsout.lon2d.attrs['units'] = 'degrees_east'
    # dsout.lat2d.attrs['long_name'] = 'Latitude'
    # dsout.lat2d.attrs['units'] = 'degrees_north'
    # dsout.W.attrs['long_name'] = 'W-component of wind'
    # dsout.W.attrs['units'] = 'm s-1'
    # dsout.REFL.attrs['long_name'] = 'Radar reflectivity (lamda = 10 cm)'
    # dsout.REFL.attrs['units'] = 'dBZ'

    # Write to netcdf file
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    dsout.to_netcdf(path=fileout, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print('Output saved: ', fileout)
