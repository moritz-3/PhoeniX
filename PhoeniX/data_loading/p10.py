# -*- coding: utf-8 -*-
"""
Collection of functions to load and handle data from the P10 beamline at DESY
"""
import numpy as np
import re
import ast
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

def load_data(filenames):
    """"
    input: list of strings
    output: np.array(3D)
    Loads data at beamline P10. Note that the detectors only allow for saving a certain amount of data.
    # Todo: check if this works for larger files.
    """
    data = []
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            next_data = f['entry/data/data']
            data.append(next_data[:])
    data = np.array(data)
    if data.ndim == 4:
        n_files, n_frames, dim_y, dim_x = data.shape
        data = data.reshape(n_files*n_frames, dim_y, dim_x)
    return data

def create_poni_from_batchinfo(scan_series, pixel_size_m=75*1e-6):
    """
    input: str, float - [*.batchinfo filename, detector pixel size (m)]
    return: pyFAI, dict - [poni-file, experimental parameters]
    Creates a poni file from the information given in batchinfo at beamline P10 (DESY).
    """
    filename_batchinfo = get_filename_batchinfo(scan_series)
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
                case 'ccdtth:':
                    ccdtth = np.deg2rad(float(tmp[-1]))
                case 'ccdx0:':
                    ccdx0 = float(tmp[1])*1e-3
                case 'ccdz0:':
                    ccdz0 = float(tmp[-1])*1e-3
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
        rot1 = 0,
        rot2 = ccdtth - ccdtth0,
        rot3 = 0,
        wavelength = wavelength_m # wavelength (m)
    )

    # create dictionary
    experimental_parameters = {}
    experimental_parameters['energy_eV'] = energy_eV
    experimental_parameters['wavelength_m'] = wavelength_m
    experimental_parameters['sample_detector_distance_m'] = sample_detector_distance_m
    experimental_parameters['direct_beam_(x,y)_pixels'] = (center_x_pixels, center_y_pixels)
    return (pyfai_config, experimental_parameters)

def get_filename_batchinfo(scan_series):
    filename_batchinfo = scan_series.sample.dir_beamtime / f'raw/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}/{scan_series.detector}/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}.batchinfo'
    return filename_batchinfo

def get_filenames_series(scan_series):
    filename_batchinfo = get_filename_batchinfo(scan_series)
    with open(filename_batchinfo, 'r') as f:
        for line in f:
            if re.search(r'\bndataend\b', line):
                tmp = line.strip()
                _, vec_str = line.split(":", 1)
                vec = ast.literal_eval(vec_str.strip())
    number_files = len(vec)
    filenames = []
    for i in range(number_files):
        filenames.append(scan_series.sample.dir_beamtime / f'raw/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}/{scan_series.detector}/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}_data_{i+1:06d}.h5')
    return filenames

def get_delay_and_exposure_time(scan_series):
    """
    input: ScanSeries
    return: float, float
    Extracts the delay time and exposure time from the master-file at P10, DESY. Note that in the case of a dark time, there is a difference between the two.
    """
    filename_master = scan_series.sample.dir_beamtime / f'raw/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}/{scan_series.detector}/{scan_series.sample.sample_name}_{scan_series.scan_number:05d}_master.h5'
    with h5py.File(filename_master, 'r') as f:
        delay_time = f['entry/instrument/detector/frame_time'][()]
        exposure_time = f['entry/instrument/detector/count_time'][()]
        dark_time = delay_time - exposure_time
        return (delay_time, exposure_time, dark_time)