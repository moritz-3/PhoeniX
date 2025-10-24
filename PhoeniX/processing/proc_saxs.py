# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import pyFAI

def process_SAXS(data2D, pyfai_config, precision_SAXS=600, mask=None):
    """
    input: np.array(2D), pyFAI-config, _int, _Mask
    return: np.array(1D), np.array(1D), np.array(1D)
    Calculates the SAXS curve (including Poisson error) from a 2D detector image. 
    The precission of data points along Q is given by 'precision_SAXS'. As standard, a value of 600 is chosen.
    Optionally, a Mask object that provides information about bad pixels can be handled by the function.
    """
    dim_y, dim_x = data2D.shape
    if (mask is None or dim_y != mask.dim_y or dim_x != mask.dim_x):
        mask_array = None
        print('No mask used')
    else:
        mask_array = np.logical_not(mask.array_boolean) # pyFAI requires the inverted mask
    # calculate SAXS curve
    SAXS_Q, SAXS_IofQ, SAXS_errorIofQ = pyfai_config.integrate1d(data = data2D, # 2D data array
        npt = precision_SAXS,
        unit='q_nm^-1', # unit of x-axis
        mask=mask_array,
        error_model = 'poisson')
    SAXS_Q = SAXS_Q*1e9
    return (SAXS_Q, SAXS_IofQ, SAXS_errorIofQ)

def process_frame_resolved_SAXS(data3D, pyfai_config, precision_SAXS=600, mask=None):
    """
    input: np.array(3D), pyFAI-config, _int, _Mask
    return: np.array(1D), np.array(2D), np.array(2D)
    Calculates the SAXS curve for each frame of a series. Thus, the input is the full dataset with dimensions (#frames, dim_y, dim_x). 
    As output, it returns the Q-values as a 1D-array (as it is the same for each frame) and 2D-arrays (dimension (#frames, #Q-bins)) for the SAXS intensity and error, respectively.
    The input precision SAXS defines the number of Q-values (the resolution).
    Q-values are given in units of 1/nm^-1.
    """
    number_frames, dim_y, dim_x = data3D.shape
    if (mask is None or dim_y != mask.dim_y or dim_x != mask.dim_x):
        mask_array = None
    else:
        mask_array = np.logical_not(mask.array_boolean) # pyFAI requires the inverted mask
    # calculate SAXS curves
    frame_resolved_SAXS_IofQ = np.zeros((number_frames, precision_SAXS))
    frame_resolved_SAXS_errorIofQ = np.zeros((number_frames, precision_SAXS))
    for idx_frame in range(number_frames):
        SAXS_Q, next_IofQ, next_errorIofQ = pyfai_config.integrate1d(data = data3D[idx_frame,:,:],
            npt = precision_SAXS,
            unit='q_nm^-1', # unit of x-axis
            mask=mask_array,
            error_model = 'poisson')
        frame_resolved_SAXS_IofQ[idx_frame,:] = next_IofQ
        frame_resolved_SAXS_errorIofQ[idx_frame,:] = next_errorIofQ
    return SAXS_Q*1e9, frame_resolved_SAXS_IofQ, frame_resolved_SAXS_errorIofQ

def evolution_SAXS(data3D, pyfai_config, precision_SAXS=600, segments=10, mask=None):
    """
    input: np.array(3D), pyFAI-config, _int, _int, _Mask
    return: np.array(1D), np.array(2D), np.array(2D)
    Calculates the SAXS curve for a series of frames. The frames are are averaged with the number of chunks defined by segments.
    As default, 10 segments are considered.
    As output, it returns the Q-values as a 1D-array (as it is the same for each frame) and 2D-arrays (dimension (#frames, #Q-bins)) for the SAXS intensity and error, respectively.
    The input precision SAXS defines the number of Q-values (the resolution).
    Q-values are given in units of 1/nm^-1.
    """
    number_frames, dim_y, dim_x = data3D.shape
    if number_frames < segments:
        raise ValueError("The number of segments exceeds the number of frames in the dataset. Consider reducing the number of segments or use function 'process_frame_resolved_SAXS' instead.")
    frames_per_segment = number_frames / segments
    if (mask is None or dim_y != mask.dim_y or dim_x != mask.dim_x):
        mask_array = None
    else:
        mask_array = np.logical_not(mask.array_boolean) # pyFAI requires the inverted mask
    # calculate SAXS curves
    evolution_SAXS_IofQ = np.zeros((segments, precision_SAXS))
    evolution_SAXS_errorIofQ = np.zeros((segments, precision_SAXS))
    for i in range(segments):
        next_data_array = np.nanmean(data3D[int(i*frames_per_segment):int((i+1)*frames_per_segment),:,:], axis=0)
        SAXS_Q, next_IofQ, next_errorIofQ = pyfai_config.integrate1d(data = next_data_array,
            npt = precision_SAXS,
            unit='q_nm^-1', # unit of x-axis
            mask=mask_array,
            error_model = 'poisson')
        evolution_SAXS_IofQ[i,:] = next_IofQ
        evolution_SAXS_errorIofQ[i,:] = next_errorIofQ
    return SAXS_Q*1e9, evolution_SAXS_IofQ, evolution_SAXS_errorIofQ
