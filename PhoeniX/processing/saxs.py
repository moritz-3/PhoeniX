# -*- coding: utf-8 -*-
"""
To Do:
Generell question: Are we ditching pyfai and try to copy the saxs integration from pygit?
Calculation for single/few frames can be optimized way more. Do this with parallising for loop
Add units where needed also directly in xarray
add meaningful status updates
"""
import numpy as np
import pyFAI
import xarray as xr
import dask.array as da
import dask


def evolution_SAXS(raw_data_3D:xr.DataArray, pyfai_config, precision_SAXS=600, segments=10, mask=None):
    """
    input: np.array(3D), pyFAI-config, _int, _int, _Mask
    return: np.array(1D), np.array(2D), np.array(2D)
    Calculates the SAXS curve for a series of frames. The frames are are averaged with the number of chunks defined by segments.
    As default, 10 segments are considered.
    As output, it returns the Q-values as a 1D-array (as it is the same for each frame) and 2D-arrays (dimension (#frames, #Q-bins)) for the SAXS intensity and error, respectively.
    The input precision SAXS defines the number of Q-values (the resolution).
    Q-values are given in units of 1/nm^-1.


    data3D xr.DataArray (raw_data)
    better scanSeries object
    even better make this function a call for ScanSeries

    input: np.array(3D), pyFAI-config, _int, _Mask
    return: np.array(1D), np.array(2D), np.array(2D)
    Calculates the SAXS curve for each frame of a series. Thus, the input is the full dataset with dimensions (#frames, dim_y, dim_x). 
    As output, it returns the Q-values as a 1D-array (as it is the same for each frame) and 2D-arrays (dimension (#frames, #Q-bins)) for the SAXS intensity and error, respectively.
    The input precision SAXS defines the number of Q-values (the resolution).
    Q-values are given in units of 1/nm^-1.

    To Do: make this @delayed with multiple calls for different slices?
    Possibility to write/copy stuff from pygit to make this faster
    GET RID OF THIS saxs evolution is enough


    Calculates the SAXS curve (including Poisson error) from a 2D detector image. 

    Parameters:
    ___________
    raw_data_3D: xr.DataArray
                Xarray with dimensions frames, y, x 
    pyfai_config : 
            pyFAI object that contains the setup parameters for q-space calculation

    precicion_SAXS : int
                     value for the amout of q-bins used for the integration

    segments : int
            nuber of final SAXS curve the measurement is dived into. The maximum is the number of frames in the scan.

    mask : None | boolean.array two dimensional
            if None: No mask
            if 2D boolean array: Pixels that are excluded from the detectorpixels as None values

    Returns:
    ___________
    xarray.Dataset :
        xarray.DataArray:
            IofQ data for each frame
        xarray.DataArray:
            IofQ error for each frame
        dimensions:
            frames
            q
        coordinates:
            q: with q values in
            max_frame_nr: number of highest frame in the segment
            max_exp_time: elapsed experimental time of the highest frame in the segment

    Info:
    ____________
    If you use this function seperated from the intended pipeline you might need a compute() call afterwards.

    To Do:
    __________
    Add unit to dataset
    better optimisation for large segment numbers
    change ScanSeries to xarray and add frame time in ScanSeries function call to be consistant with xpcs functions
    """
    number_frames, dim_y, dim_x = raw_data_3D.shape
    if number_frames < segments:
        raise ValueError("The number of segments exceeds the number of frames in the dataset. Consider reducing the number of segments or use function 'process_frame_resolved_SAXS' instead.")
    frames_per_segment = number_frames / segments
    if (mask is None or dim_y != mask.dim_y or dim_x != mask.dim_x):
        mask_array = None
    else:
        mask_array = np.logical_not(mask.array_boolean) # pyFAI requires the inverted mask

    # calculate SAXS curves
    evolution_SAXS_IofQ = xr.DataArray(data=da.zeros((segments, precision_SAXS)), #leave this as numpy! dask does not make it faster and leads to problems later
                                       dims=['frames','q'],
                                       )
    evolution_SAXS_errorIofQ = xr.DataArray(data=da.zeros((segments, precision_SAXS)), #leave this as numpy! dask does not make it faster and leads to problems later
                                       dims=['frames','q'],
                                       )


    for i in range(segments):
        next_data_array = raw_data_3D.sel(frames=slice(int(i*frames_per_segment),int((i+1)*frames_per_segment))).mean(dim='frames', skipna=True)
        SAXS_Q, next_IofQ, next_errorIofQ = pyfai_config.integrate1d(data = da.from_array(next_data_array),
            npt = precision_SAXS,
            unit='q_nm^-1', # unit of x-axis
            mask=mask_array,
            error_model = 'poisson')
        evolution_SAXS_IofQ[i] = next_IofQ
        evolution_SAXS_errorIofQ[i] = next_errorIofQ

    evolution_SAXS_results = xr.Dataset(data_vars={'IofQ' : evolution_SAXS_IofQ,
                                                    'IofQerror' : evolution_SAXS_errorIofQ,
                                                    },
                                        attrs={'segments':segments,
                                                'frames_per_segment':int(frames_per_segment),
                                                }
                                        )
    evolution_SAXS_results = evolution_SAXS_results.assign_coords(q = ('q', SAXS_Q*1e9))
    evolution_SAXS_results = evolution_SAXS_results.assign_coords(max_frame_nr = ('frames', [(x+1)*frames_per_segment for x in range(evolution_SAXS_results['frames'].shape[0])]))

    return evolution_SAXS_results

