# -*- coding: utf-8 -*-
"""
Collection of functions to load and handle data from the P10 beamline at DESY

To Do: Check if exposure and delay-/dark times are important correctly and how to implement here and further
        They should yield a general single frame-time that is valid for all further computations
        Implement different x- y-pixel sizes
        Data logger / save for pyfai config
"""
import scipy
# pyFAI (version control)
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator # since 2025
except ImportError:
    try:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator # till ~2022
    except ImportError:
        from pyFAI import AzimuthalIntegrator
# end pyFAI (version control)
import h5py
import xarray as xr
from pathlib import Path
import dask.array as da
import numpy as np

def get_filename_batchinfo(scan_series:ScanSeries) -> Path:
    """"
    Parameters:
    ___________
    scan_series : str
                    name of the scan series / experiment (*.batchinfo)

    Returns:
    ___________
    path to batchinfo file : Path
    """
    filename_batchinfo:Path = scan_series.sample.dir_beamtime / f'raw/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}/{scan_series.detector}/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}.batchinfo'
    return filename_batchinfo


def get_filename_master_h5(scan_series:ScanSeries) -> Path:
    """"
    Parameters:
    ___________
    scan_series : str
                    name of the scan series / experiment (*_master.h5)

    Returns:
    ___________
    path to master h5 file : Path
    """

    filename_master_h5:Path = scan_series.sample.dir_beamtime / f'raw/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}/{scan_series.detector}/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}_master.h5'
    return filename_master_h5



def load_data(scan_series:ScanSeries, chunk_size:str|tuple[int:int:int]='auto'): 
    """"
    loads detector images from h5 file(s) into an xarray
    adds time axis to frame number in the xarray
    adds raw_data (xarray with dimensions frames, y, x and coordinates frame_nr and exp_time) to ScanSeries Object

    Parameters:
    ___________
    scan_series : ScanSeries
                    Object that unifies experimental parameters (see general/scans.py ScanSeries)

    chunk_size : str | tuple[int:int:int]
                chunk size of the blocks the data is loaded into memory can be 'auto' or (int,int,int)
                Fastest computation for SAXS should look like this (int, -1, -1), -1 is the whole axis so this would leave each detector image intact and would leave variable size of frames loaded into memory at the same time

    To Do:
    ___________
    # Check frame time and count time difference?
    """

    filename = get_filename_master_h5(scan_series)

    with h5py.File(filename, 'r') as f:
        data_keys = [key for key in f['entry/data/'].keys()]
        frame_time = f['entry/instrument/detector/frame_time']
        exposure_time = f['entry/instrument/detector/count_time'][()]
        scan_series.add_frame_time(frame_time[()])
        scan_series.add_exposure_time(exposure_time[()])

        for i, key in enumerate(data_keys):
            match i:
                case 0:
                    data_array=da.from_array(f[f'entry/data/{key}'],
                                  chunks=chunk_size,
                                  )

                    data = xr.DataArray(data_array,
                                        dims=("frames", "y", "x"),
                                        name='raw_Data',
                                        )

                case _:
                    data_array=da.from_array(f[f'entry/data/{key}'],
                                  chunks=chunk_size,
                                  )
                    next_data = xr.DataArray(data_array,
                                             dims=("frames", "y", "x"),
                                             )
                    data = xr.concat([data, next_data],
                                     dim = 'frames'
                                     )


    data = data.assign_coords(frame_nr = ('frames', [x+1 for x in range(data['frames'].shape[0])]))
    data = data.assign_coords(exp_time = ('frames', [t*scan_series.frame_time for t in range(data['frames'].shape[0])])) #check if frame_time is enoug or if more complicated with dark times
    scan_series.add_raw_data(data)

def create_poni_file(scan_series:ScanSeries):# ->  pyFAI.azimuthalIntegrator.AzimuthalIntegrator | pyFAI.integrator.common.Integrator , dict[str:Any]:
    #define poni type
    """"
    creates a pyfai poni file and a dictionary with experimental parameters form ScanSeries Object
    adds the pyfai object to the ScanSeries Object




    Parameters:
    ___________

    str, float - [*.batchinfo filename, detector pixel size (m)]
    scan_series : str
                    name of the scan series / experiment (*.batchinfo)


    Returns:
    ___________
    pyFAI poni-file :

    experimental_parameters: dict[str:Any]
                             Dictionary containing beamenergy, beam wavelenght, sample detector distance and beamcenter


    To Do:
    ___________
    # Todo: Check if pixel size is needed or if better to read from h5 file

    """
    filename_batchinfo = get_filename_batchinfo(scan_series)
    filename_master = get_filename_master_h5(scan_series)

    with h5py.File(filename_master, 'r') as f:
        pixel_size_m = f['entry/instrument/detector/x_pixel_size'][()] # x and y pixel size are equal

    with open(filename_batchinfo, 'r') as f:
        for line in f:
            tmp = line.strip().split()
            match tmp[0]:
                case 'energy:':
                    energy_eV = float(tmp[-1])*1e3
                case 'rr:':
                    sample_detector_distance_m = float(tmp[-1])*1e-3
                case 'x0:':
                    x0 = int(tmp[-1])
                case 'y0:':
                    y0 = int(tmp[-1])
                case 'ccdx:':
                    ccdx = float(tmp[1])*1e-3
                case 'ccdz:':
                    ccdz = float(tmp[1])*1e-3
                case 'ccdx0:':
                    ccdx0 = float(tmp[1])*1e-3
                case 'ccdz0:':
                    ccdz0 = float(tmp[-1])*1e-3
                case 'ccdtth:':
                    ccdtth = np.deg2rad(float(tmp[-1]))
                case 'ccdtth0:':
                    ccdtth0 = np.deg2rad(float(tmp[-1]))

    center_x_pixels = x0 + (ccdx - ccdx0) / pixel_size_m
    center_y_pixels = y0 + (ccdz - ccdz0) / pixel_size_m
    wavelength_m = scipy.constants.h * scipy.constants.speed_of_light / (energy_eV * scipy.constants.e)
    # create poni file
    pyfai_config = AzimuthalIntegrator(
        dist = sample_detector_distance_m, # distance (m)
        pixel1 = pixel_size_m, # pixel size (m) in y-direction
        pixel2 = pixel_size_m, # pixel size (m) in x-direction
        poni1 = center_y_pixels * pixel_size_m, # position of direct beam (m) in y-direction
        poni2 = center_x_pixels * pixel_size_m, # position of direct beam (m) in x-direction
        rot1 = 0, # rotation of detector (ToDo)
        rot2 = ccdtth - ccdtth0,
        rot3 = 0,
        wavelength = wavelength_m # wavelength (m)
        )

    scan_series.add_pyfai_config(pyfai_config)
