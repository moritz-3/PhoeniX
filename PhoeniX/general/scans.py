# -*- coding: utf-8 -*-
"""

"""
from pathlip import Path
import numpy as np
import autoXPCS.processing.xpcs as proc_xpcs

class Sample:
    """
    class Sample contains the info about the beamtime(Path) and the sample name (str).
    """
    def __init__(self, dir_beamtime, sample_name):
        self.dir_beamtime = Path(dir_beamtime)
        self.sample_name = sample_name
        self.info = {}
        self.info['dir_beamtime'] = str(self.dir_beamtime)
        self.info['sample_name'] = str(self.sample_name)

class ScanCollection:
    """
    At P10, different types of XPCS-scans exist, namely 'series', 'dscan' and 'meshscan'. 
    The later two contain multiple series of scans, all aquired with the same parameters except for one (dscan) or two (meshscan) motors (e.g., motor 'samx' to scan the sample in x-direction).
    This allows for example to perform position resolved XPCS, but also is of advantage to average XPCS data of identical scans of a given sample. 
    While series is defined separately, ScanCollection allows to perform the same type of analysis for all scans of a dscan/meshscan.

    Attributes
    ----------
    sample : Sample
        Sample object that contains the directory of the beamtime and the sample name
    detector : str
        Detector abbreviation, used for loading data (in the case of P10, DESY) but also used to distinguish between different detectors that record data at the same time.
    scan_number : int
        Specifies the sample in combination with the sample name.
    skip_frames_start=0 : int
        At P10, the shutter might open over the cause of the first few frames recorded. Thus, these frames contain data that is not of use for XPCS analysis. 
        Note that the shutter speed at P10 is approximately 13 ms. Thus, especially scans with an exposure time shorter or equal to 13 ms suffer significantly from the opening of the shutter on the first frame(s).
    scan : Scan
        Scan object. The Scan object presents one of the spots 

    """
    def __init__(self, sample, detector, scan_number, skip_frames_start=0):
        self.sample = sample
        self.detector = detector
        self.scan_number = scan_number
        self.skip_frames_start = skip_frames_start
        self.scan = None
        self.counter_scan_series = 0

    def load_scan(self, scan, counter_scan_series = None):
        self.scan = scan
        if counter_scan_series is None:
            self.counter_scan_series = counter_scan_series + 1
        else:
            self.counter_scan_series = counter_scan_series
            
class ScanSeries:
    """
    """
    def __init__(self, sample, detector, scan_number, skip_frames_start=0):
        self.sample = sample
        self.detector = detector
        self.scan_number  = scan_number
        self.skip_frames_start = skip_frames_start
        self.filenames_raw_data = []
        self.raw_data = None
        self.number_frames = None
        self.dim_x = None
        self.dim_y = None
        self.experimental_parameters = {}
        self.delay_time_s = None
        self.exposure_time_s = None
        self.mask = None
        self.input = {}
        self.input['sample'] = sample.info
        self.input['detector'] = detector
        self.input['scan_number'] = scan_number
        self.input['skip_frames_start'] = skip_frames_start

    def add_mask(self, mask):
        """
        input: mask
        Add Mask-object as parameter to a Scan-object. Update mask to info.
        """
        self.mask = mask
        self.input['mask'] = mask.create_dict_output()

    def add_filenames_raw_data(self, filenames_raw_data):
        self.filenames_raw_data = filenames_raw_data
        self.input['filenames_raw'] = filenames_raw_data

    def add_raw_data(self, raw_data):
        self.raw_data = raw_data
        self.number_frames, self.dim_y, self.dim_x = raw_data.shape
        self.input['number_frames'] = self.number_frames
        self.input['dim_y'] = self.dim_y
        self.input['dim_x'] = self.dim_x

    def add_delay_time(self, delay_time, exposure_time=None):
        """
        input: float, _float
        Adds the parameters 'delay_time' and 'exposure_time' to the ScanSeries object. 
        These two parameters can differ in case of a dark time as is the case at P10, DESY. If they are the same, only 'delay_time' needs to be provided as input.
        """
        self.delay_time = delay_time
        if exposure_time is None:
            self.exposure_time = delay_time
        else:
            self.exposure_time = exposure_time

    def calculate_ttcfs(self, q_partitions):
        """
        input: QPartitions
        return: np.array(2D), np.array(3D), np.array(1D)
        Calculates the two-time correlation function for each partition provided by 'q_partitions' for the ScanSeries.
        """
        data_3D = self.raw_data[self.skip_frames_start:,:,:]
        number_frames_analyzed = self.number_frames - self.skip_frames_start
        # initialize dummy arrays
        intensities_per_q_partition = np.zeros((q_partitions.number_q_partitions, number_frames_analyzed))
        pixels_per_q_partition = np.zeros(q_partitions.number_q_partitions)
        ttc_per_q_partition = np.zeros((q_partitions.number_q_partitions, number_frames_analyzed, number_frames_analyzed))
        # calculate ttc for each partition
        for i in range(q_partitions.number_q_partitions):
            # combine general mask and q-partition
            if self.mask is None:
                next_mask = q_partitions.arrays[i]
            else:
                next_mask = np.logical_and(q_partitions.arrays[i,:,:],self.mask.array_boolean)
            # reduce dataset from 3D (number_frames, dim_y, dim_x) to 2D (number_frames, dim_xy) with dim_xy determined by the mask
            next_data_2D = data_3D[:,next_mask]
            # calculate ttc
            next_intensity_per_q_partition, next_number_of_pixels, next_ttc = proc_xpcs.calculate_ttcf(next_data_2D)
            intensities_per_q_partition[i,:] = next_intensity_per_q_partition
            pixels_per_q_partition[i] = next_number_of_pixels
            ttc_per_q_partition[i,:,:] = next_ttc
        return (intensities_per_q_partition, ttc_per_q_partition, pixels_per_q_partition)

    def calculate_all_g2s(self, ttcfs):
        number_frames_analyzed = self.number_frames - self.skip_frames_start
        number_partitions, number_frames_ttc, _ = ttcfs.shape
        if number_frames_ttc != number_frames_analyzed:
            raise ValueError("Attention, dimensions according to ScanSeries do not match with ttcf size.")
        g2_functions = np.zeros((number_partitions,number_frames_analyzed - 1)) # omit self-correlation
        for i in range(number_partitions):
            g2_functions[i,:] = proc_xpcs.calculate_g2(ttcfs[i])
        return g2_functions

    def calculate_g2s_multitau(self, q_partitions, layers_per_level=4):
        """
        """
        data_3D = self.raw_data[self.skip_frames_start:,:,:]
        number_frames_analyzed = self.number_frames - self.skip_frames_start
        g2s_multitau = []
        g2errors_multitau = []
        # calculate g2s_multitau for each partition
        for i in range(q_partitions.number_q_partitions):
            # combine general mask and q-partition
            if self.mask is None:
                next_mask = q_partitions.arrays[i]
            else:
                next_mask = np.logical_and(q_partitions.arrays[i,:,:],self.mask.array_boolean)
            # reduce dataset from 3D (number_frames, dim_y, dim_x) to 2D (number_frames, dim_xy) with dim_xy determined by the mask
            next_data_2D = data_3D[:,next_mask]
            times_multitau, next_g2_multitau, next_g2error_multitau = proc_xpcs.multitau(next_data_2D, layers_per_level=4)
            g2s_multitau.append(next_g2_multitau)
            g2errors_multitau.append(next_g2error_multitau)
        return (times_multitau, np.array(g2s_multitau), np.array(g2errors_multitau))

    def add_input(self, input_dict, specifier=None):
        if specifier is None:
            self.input.update(input_dict)
        else:
            self.input[specifier] = input_dict
        
        
###############################################################################
#####                       functions for saving                          #####
###############################################################################
def save_incoherent_image_results(dir_save, incoherent_img, incoherent_img_cor=None):
    """
    input: str/Path, np.array(2D), _np.array(2D)
    return: dict
    Saves the results of the incoherent image to the folder provided. 
    """
    summary_results = {}
    # incoherent image
    dim_y, dim_x = incoherent_img.shape
    summary_incoherent_img = {'filename': str(dir_save) + '/incoherent_img.npy',
                             'dim_y': dim_y,
                             'dim_x': dim_x}
    summary_results['incoherent_img'] = summary_incoherent_img
    np.save(Path(dir_save) / 'incoherent_img.npy', incoherent_img)
    if incoherent_img_cor is not None:
        dim_y, dim_x = incoherent_img_cor.shape
        summary_incoherent_img_cor = {'filename': str(dir_save) + '/incoherent_img_cor.npy',
                                     'dim_y': dim_y,
                                     'dim_x': dim_x}
        summary_results['incoherent_img_cor'] = summary_incoherent_img_cor
        np.save(Path(dir_save) / 'incoherent_img_cor.npy', incoherent_img_cor)
    return summary_results

def save_SAXS_results(dir_save, filename_save_general, SAXS_Q, SAXS_IofQ, SAXS_errorIofQ=None):
    """
    input: str/Path, str, np.array(1D), np.array(1/2D), _np.array(1/2D)
    return: dict
    ToDO
    """
    summary_results = {}
    precision = len(SAXS_Q)
    np.save(Path(dir_save) / f'{filename_save_general}_Q.npy', SAXS_Q)
    if SAXS_IofQ.ndim == 1:
        number_curves = 1
    else:
        number_curves = SAXS_IofQ.shape[0]
    summary_results = {'precision': precision,
                       'number_curves': number_curves,
                       'Q': str(dir_save) + f'{filename_save_general}_Q.npy',
                       'IofQ': str(dir_save) + f'{filename_save_general}_IofQ.npy'}
    if SAXS_errorIofQ is None:
        summary_results['description'] = 'Dimension of IofQ is (number_curves, precision).'
        np.save(Path(dir_save) / f'{filename_save_general}_IofQ.npy', SAXS_IofQ)
    else:
        summary_results['description'] = 'Dimension of IofQ is (2, number_curves, precision), where IofQ[0,:,:] contains IofQ and IofQ[1,:,:] contains errorIofQ.'
        data_save = np.stack((SAXS_IofQ, SAXS_errorIofQ), axis=0)
        np.save(Path(dir_save) / f'{filename_save_general}_IofQ.npy', data_save)
    return summary_results  

def save_Q_partitions(dir_save, q_partitions):
    """
    input: str/Path, QPartitions
    return: dict
    """
    summary_results = q_partitions.info
    summary_results['filename'] = str(dir_save) + '/q_partitions.npy'
    np.save(Path(dir_save) / 'q_partitions.npy', q_partitions.arrays)
    return summary_results

def save_results_TTCF(dir_save, intensities_per_partition, ttc_per_partition, pixels_per_q_partition):
    """
    input:
    return: dict
    """
    summary_results = {'intensities_per_partition': str(dir_save) + '/intensities_per_partition.npy',
                      'ttc_per_partition': str(dir_save) + '/ttc_per_partition.npy',
                      'pixels_per_partition': pixels_per_q_partition,
                      'description': 'intensities: (#partitions,#frames); ttc: (#partitions,#frames,#frames)'}
    np.save(Path(dir_save) / 'intensities_per_partition.npy', intensities_per_partition)
    np.save(Path(dir_save) / 'ttc_per_partition.npy', ttc_per_partition)
    return summary_results

def save_results_g2(dir_save, filename_save_general, times, g2s):
    """
    input:
    return: dict
    """
    summary_results = {'filename_times': str(dir_save) + f'/{filename_save_general}_times.npy',
                      'filename_g2s': str(dir_save) + f'/{filename_save_general}_g2s.npy'}
    np.save(Path(dir_save) / f'{filename_save_general}_times.npy', times)
    np.save(Path(dir_save) / f'{filename_save_general}_g2s.npy', g2s)
    return summary_results