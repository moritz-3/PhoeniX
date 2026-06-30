# -*- coding: utf-8 -*-
"""
ToDo:   check if returned ttcfs are correct
        resolve memory issue for multiple large Q-rings
        implement meaningful status updates
"""

import numpy as np
import xarray as xr
import dask.array as da

def calculate_ttcfs(data_3D:xr.DataArray, q_rings:QRings, mask=None, include_ttcf_via_std=False):
    """
    return: np.array(2D), np.array(3D), np.array(1D)
    Calculates the two-time correlation function for each partition provided by 'q_partitions' for the ScanSeries.
    mask: 2Darray

    Info: if you use this system away from the inteded pipeline you might need a compute() call on it.
    """

    number_frames_analyzed = len(data_3D['frames'])
    number_of_q_partitions = len(q_rings.mask_arrays['partitions'])

    # initialize dummy arrays
    pixels_per_q_partition = da.zeros(number_of_q_partitions)
    ttc_per_q_partition = da.zeros((number_of_q_partitions, number_frames_analyzed, number_frames_analyzed))
    if include_ttcf_via_std == True:
        std_ttc_per_q_partition = da.zeros((number_of_q_partitions, number_frames_analyzed, number_frames_analyzed))

    # calculate ttc for each partition
    for i in range(number_of_q_partitions):

        # combine general mask and q-partition
        next_mask = ttcf_qring_masking(q_rings.mask_arrays[i],mask=mask)

        # reduce dataset from 3D (number_frames, dim_y, dim_x) to 2D (number_frames, dim_xy) with dim_xy determined by the mask
        next_data_2D = data_3D.where(next_mask!=0,np.nan).stack(xy=('x','y')).dropna(dim='xy').compute()

        print(i)
        # calculate ttc
        if include_ttcf_via_std==False:
            next_number_of_pixels, next_ttc = ttcf(next_data_2D, include_ttcf_via_std=include_ttcf_via_std)
        if include_ttcf_via_std==True:
            next_number_of_pixels, next_ttc, next_std_ttc = ttcf(next_data_2D, include_ttcf_via_std=include_ttcf_via_std)
            std_ttc_per_q_partition[i,:,:] = next_std_ttc

        pixels_per_q_partition[i] = next_number_of_pixels
        ttc_per_q_partition[i,:,:] = next_ttc

    ttcfs = xr.Dataset(data_vars={'ttcfs': (('q','frame1','frame2'),ttc_per_q_partition),
                                  'pixel_in_ring': ('q',pixels_per_q_partition),
                                  'q_widths':q_rings.q_widths,
                                  },
                       coords={'q':q_rings.q_centers},
                       )

    if include_ttcf_via_std==True:
        ttcfs_std = xr.Dataset(data_vars={'ttcfs_std': (('q','frame1','frame2'),std_ttc_per_q_partition),
                                      'pixel_in_ring': ('q',pixels_per_q_partition),
                                      'q_widths':q_rings.q_widths,
                                      },
                           coords={'q':q_rings.q_centers},
                           )
        return ttcfs, ttcfs_std

    return ttcfs

def ttcf_qring_masking(qring_mask:xr.DataArray, mask:None|da.array=None) -> da.Array:
        if mask is None:
            full_mask = qrings_mask
        else:
            full_mask = da.logical_and(qring_mask, mask)
        return full_mask

def ttcf(data_2D,include_ttcf_via_std=None):
    """
    input: np.array(2D) - (#frames, #pixels_xy)
    include_ttcf_via_std if True TTC with STD will be calculated in addition
    return if include_ttcf_via_std=None: np.array(#frames), int, np.array(#frames, #frames)
    return if include_ttcf_via_std=True: np.array(#frames), int, np.array(#frames, #frames), np.array(#frames, #frames)
    Calculates the two-time correlation function (TTCF) for a 2D np.array with dimensions (#frames, #pixels_xy). 
    This data-format can be retrieved from the standard raw-dataset of an XPCS experiment (dimensions (#frames, dim_y, dim_x)) via ToDO
    """

    # ensure that data_2D is of dtype float32 (helps to speed up calculations)
    if data_2D.dtype == 'float32':
        data = data_2D
    else:
        data = data_2D.astype('float32')

    # calculate variation in intensity per frame
    intensity_change = da.from_array(data_2D.mean(dim='xy',skipna=True))
    if include_ttcf_via_std == True:
        #std intensity per frame
        std = da.from_array(data_2D.std(dim='xy',skipna=True))# in principle mean=intensity_change should prevent recalulating the mean but does not work maybe manuel std with privious caclulated mean is faster?
    # determine number of pixels
    number_of_pixels = data_2D.xy.shape[0]
    # calculate ttc via dot-product
    ttc = da.dot(data, data.transpose()) #use da.dot xr.dot results in empty array
    # normalisation of ttc
    ttc_norm = ttc / da.outer(intensity_change, intensity_change) / number_of_pixels
    if include_ttcf_via_std == True:
        ttc_std_numerator = ttc/ number_of_pixels - da.outer(intensity_change, intensity_change)
        ttc_std_denominator = da.outer(std,std) #add this line with condition (maybe all three lines needed)
        ttc_std = ttc_std_numerator/ttc_std_denominator

        return number_of_pixels, ttc_norm, ttc_std
    else:
        return number_of_pixels, ttc_norm


# from here on old functions
def calculate_g2(ttcf):
    """ 
    input: np.array(2D)
    return: np.array(1D)
    Calculates the autocorrelation function as diagonal cuts from the two-time correlation function.
    According to [O. Bikondoa, J. Appl. Cryst. (2017), 50, 357-68], these cuts are referred to 'alternative coordinate system' (ACS).   
    """
    g2 = np.array([np.nanmean(np.diagonal(ttcf, offset=i)) for i in range(1,(ttcf.shape[0]))])
    return g2

def perform_diagonal_cut4ttcf(ttcf, idx_center, idx_width):
    """
    input: np.array(2D)
    return: np.array(1D)
    """
    dim_steps, _ = ttcf.shape
    included_data = np.full((dim_steps, dim_steps), np.nan)
    idx_min = np.max([0,int(idx_center - idx_width/2)])
    idx_max = np.min([dim_steps,int(idx_center + idx_width/2)]) 
    for i in range(idx_min, idx_max):
        np.fill_diagonal(included_data[:, i:], 1)
    included_data = np.flipud(included_data.transpose())
    g2_diagonal_cut = calculate_g2(np.multiply(ttcf,included_data))
    return g2_diagonal_cut

def perform_horizontal_cut4ttcf(ttcf, idx_center, idx_width):
    """
    """
    dim_steps, _ = ttcf.shape
    included_data = np.full((dim_steps, dim_steps), np.nan)
    idx_min = np.max([0,int(idx_center - idx_width/2)])
    idx_max = np.min([dim_steps,int(idx_center + idx_width/2)])
    included_data[idx_min:idx_max,:] = 1 # note that this extends over the diagonal, but the function calculate_g2 only calculates everything below the diagonal    
    g2_horizontal_cut = calculate_g2(np.multiply(ttcf,included_data))
    return g2_horizontal_cut

def perform_vertical_cut4ttcf(ttcf, idx_center, idx_width):
    """
    """
    dim_steps, _ = ttcf.shape
    included_data = np.full((dim_steps, dim_steps), np.nan)
    idx_min = np.max([0,int(idx_center - idx_width/2)])
    idx_max = np.min([dim_steps,int(idx_center + idx_width/2)])
    included_data[:,idx_min:idx_max] = 1 # note that this extends over the diagonal, but the function calculate_g2 only calculates everything below the diagonal    
    g2_vertical_cut = calculate_g2(np.multiply(ttcf,included_data))
    return g2_vertical_cut

### These functions evolve around the multitau algorithm

def get_delay_info(numberframes, numberdelaysperlevel=4):
    """
    input: int, _int - [#frames, #delays per level]
    return: (np.array(1D), np.array(1D)) - [array of indices, array of levels]
    Calculates the timesteps in units of frames for the different levels of the multitau-algorithm. The second output is an array of the level at each timestep. 
    This level indicates over how many frames the average is taken before the correlation in form of a dot product is calculated.
    E.g., level=0, average over 2^0=1 frames; level=1, average over 2^1=2 frames; level=2, average over 2^2=4 frames; ...
    As default, the number of delays per level is set to 4. 
    To better account for the information given by the early delays, level 0 contains 2*(#delays per level) - 1 -> omit the selfcorrelation at zero delay.
    """
    counter_levels = -1 # counts at what level (0,1,2,...) for the power to base 2 we are (2^0=1, 2^1=2, 2^2=4, ...). Give an additional level -1 at the beginning which counts in integers
    delay_starting_indices = []
    level_per_index = []
    modulo_counter = 1
    next_index = 0
    while next_index < numberframes-np.max((1,2**counter_levels)): # terminate if the next index would exceed numberframes, further ensure that the last entry contains the maximal number of frames so that noise is the lowest
        delay_starting_indices.append(next_index) # add another index
        if modulo_counter == numberdelaysperlevel:
            modulo_counter = 1
            counter_levels += 1
        else:
            modulo_counter += 1
        next_index += np.max((1, 2**counter_levels))
        level_per_index.append(np.max((0,counter_levels)))
    return (np.array(delay_starting_indices, dtype='int')+1, np.array(level_per_index))

def block_average_rows(A,k):
    dim_x, dim_y = np.shape(A)
    number_blocks = int(np.floor(dim_x/k)) # attention, as the last block would contain less frames than the previous, it is omitted
    B = np.zeros((number_blocks, dim_y))

    for i in range(number_blocks):
        next_block = np.arange(i*k,(i+1)*k)
        B[i,:] = np.mean(A[next_block,:], axis=0)

    return B

def multitau(data2D, time_interval=1, layers_per_level=4):
    """
    input: np.array(2D), float, int - [data (dim_#frames, dim_#pixels_xy), time interval in seconds, layers per level in multitau-algorithm]
    return: (np.array(1D), np.array(1D), np.array(1D)) - [delay times, g2, error g2]
    Calculates the autocorrelation function g2 on the basis of a multitau-algorithm. 
    Delay-times are calculated based on the time interval, the number of frames in 2Ddata and the layers per level, making use of function 'getdelayinfo.
    """
    number_frames, number_pixels = data2D.shape
    starting_indices, levels_per_index = get_delay_info(number_frames, layers_per_level)
    number_times_multitau = len(levels_per_index)
    times_multitau = time_interval * starting_indices
    g2 = np.zeros(number_times_multitau)
    g2_error = np.zeros(number_times_multitau)
    # at each level, the next delay is calculated starting from layers_per_level+1 for the correlations.
    current_level = 0
    current_data2D = data2D
    delay_index_within_level = 1
    for i in range(number_times_multitau):
        if levels_per_index[i] > current_level:
            current_level += 1
            current_data2D = block_average_rows(current_data2D,2)
            delay_index_within_level = layers_per_level
        counts_past = current_data2D[:-delay_index_within_level,:].astype(np.float32)
        counts_future = current_data2D[delay_index_within_level:,:].astype(np.float32)
        correlation = counts_past * counts_future # elementwise multiplication averaged over all pixels
        frame_correlation = np.mean(correlation, axis=1) / (np.mean(counts_past, axis=1)*np.mean(counts_future, axis=1))
        g2[i] = np.mean(frame_correlation)
        g2_error[i] = np.std(frame_correlation, ddof=1) / np.sqrt(len(frame_correlation))
        delay_index_within_level += 1

    return(times_multitau, g2, g2_error)
