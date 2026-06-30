# -*- coding: utf-8 -*-
"""
To Do:  Get rid of all the dictionary inits and move this to save functions
        ScanSeries should take care of skip first frames
        implement skip last frames
        add none values to ScanSeries to check and give better errors
        saxs and xpcs functions are not consistant in e.g. mask)
        add saxs function calls for average obver all frames and saxs for each individual frame
"""
from pathlib import Path
import numpy as np
from processing import saxs, xpcs

class Sample:
    """
    Contains gneneral information about the sample, specifically the sample name and the path to sample folder

    Attributes:
    ___________

    sample_name : str
                the name of the folder that was given for the sample during the experiment

    dir_beamtime : pathlib.Path
                path to the folder of the beamtime during which the sample was measured
    """
    def __init__(self, dir_beamtime:Path, sample_name:str):
        self.sample_name = sample_name
        self.dir_beamtime = dir_beamtime
        """
        better to make a function with this for saving?
        self.info = {}
        self.info['dir_beamtime'] = str(self.dir_beamtime)
        self.info['sample_name'] = str(self.sample_name)
        """


class ScanSeries:
    """
    Contains information and data of the specific scan series

    Attributes:
    ___________

    sample : Sample
            Sample object that contains the sample name (folder name of the experiment) and the path to the beamtime folder

    detector : str
            Name of the detector used for the data aquesition

    scan_number : int
            Number of the scan (folder) the specific measurement is saved in

    skip_frames_start : int
            Number of first frames to skip in the calculations e.g. to eliminate shutter effects

    raw_data : xr.DataArray
            Raw data from the detector with dimensions frames, y and x. Addionally, experimental time(exp_time) and frame number (frame_nr) is defined as coordinates along frames dimension

    number_frames : int
            The number of frames that were taken during the aquestition

    dim_x : int
            Number of pixels in the detector x-dimension

    dim_y : int
            Number of pixels in the detector y-dimension

    frame_time : float
            time between two images

    mask : masking.Mask

    temperature : float | list
        temperature(s) during the scan

    Methods:
    ___________

    add_mask(mask:maskin.Mask)
        adds Mask object with information on what pixels to exclude


    add_raw_data(raw_data:xr.DataArray)
        adds raw data of the scan series

    """
    def __init__(self, sample:Sample, detector:str, scan_number:int, skip_frames_start:int|None=None, skip_frames_end:int|None=None):
        self.sample:Sample = sample
        self.detector:str = detector
        self.scan_number:int  = scan_number
        self.skip_frames_start:int|None = skip_frames_start
        self.skip_frames_end:int|None = skip_frames_end
        self.raw_data = None
        self.number_frames = None
        self.dim_x = None
        self.dim_y = None
        #self.experimental_parameters = {}
        self.frame_time = None
        #self.delay_time_s = None
        #self.exposure_time_s = None
        self.mask = None
        self.temperature = None

        """
        better to make a function with this for saving?
        self.input = {}
        self.input['sample'] = sample.info
        self.input['detector'] = detector
        self.input['scan_number'] = scan_number
        self.input['skip_frames_start'] = skip_frames_start
        """

    def add_mask(self, mask):
        """
        input: mask
        Add Mask-object as parameter to a Scan-object. Update mask to info.
        """
        self.mask = mask
        #self.input['mask'] = mask.create_dict_output()

    def add_filenames_raw_data(self, filenames_raw_data):
        #not needed?
        self.filenames_raw_data = filenames_raw_data
        #self.input['filenames_raw'] = filenames_raw_data

    def add_raw_data(self, raw_data:xr.DataArray):
        """
        raw_data xr.DataArray must be an xarray with dimensions frames, y and x. Addionally experimental time is defined as coordinates along frames dimension
        """
        self.raw_data:xr.DataArray = raw_data # in principell skip frames cult be implemented here. Decicion to put in in calculation makes it more useabal
        self.number_frames = len(raw_data['frames'])
        self.dim_y = len(raw_data['y'])
        self.dim_x = len(raw_data['x'])
        #self.input['number_frames'] = self.number_frames
        #self.input['dim_y'] = self.dim_y
        #self.input['dim_x'] = self.dim_x

    def slice_first_to_last_frame(self):
        """
        reduces the data to the data excluding the frames skipped in skip_frames_start and skip_frames_end
        """

        if self.skip_frames_start == None and self.skip_frames_end == None: return self.raw_data

        elif self.skip_frames_end == None:
            if self.skip_frames_start > self.number_frames:
                raise IndexError(f'The number of skipped frames exceeds the number of frames in the measurement, which is {self.number_frames}.')
            return self.raw_data.sel(frames=slice(self.skip_frames_start,self.number_frames))

        elif self.skip_frames_start == None:
            if self.skip_frames_end > self.number_frames:
                raise IndexError(f'The number of skipped frames exceeds the number of frames in the measurement, which is {self.number_frames}.')
            return self.raw_data.sel(frames=slice(self.skip_frames_start,self.number_frames-self.skip_frames_end))

        else:
            if self.skip_frames_start+self.skip_frames_end > self.number_frames:
                raise IndexError(f'The number of skipped frames exceeds the number of frames in the measurement, which is {self.number_frames}.')
            return self.raw_data.sel(frames=slice(self.skip_frames_start,self.number_frames-self.skip_frames_end))

    def add_frame_time(self, frame_time):
        """
        input: float, _float
        Adds the parameters frame_time
        The frame time is the complete time between two frames.
        """
        self.frame_time = frame_time

    def add_exposure_time(self, exposure_time):
        """
        input: float, _float
        Adds the parameters 'exposure_time' to the ScanSeries object. 
        The exposure time is the time the samplis exposed to the x-ray
        """
        self.exposure_time = exposure_time


    def add_pyfai_config(self, pyfai_config):
        self.pyfai_config = pyfai_config

    def calculate_saxs_evo(self, precision_SAXS=600, segments=10):
        raw_data_slice = self.slice_first_to_last_frame()
        saxs_evo_results = saxs.evolution_SAXS(raw_data_slice, self.pyfai_config, precision_SAXS, segments, mask=self.mask)

        saxs_evo_results = saxs_evo_results.assign_coords(max_exp_time = ('frames', [(t+1)*saxs_evo_results.frames_per_segment*self.frame_time for t in range(saxs_evo_results['frames'].shape[0])])) #check if frame_time is enoug or if more complicated with dark times
        saxs_evo_results = saxs_evo_results.compute()
        self.saxs_evo_results = saxs_evo_results

    def calculate_saxs_mean(self, precision_SAXS=600):
        raw_data_slice = self.slice_first_to_last_frame()
        saxs_evo_results = saxs.evolution_SAXS(raw_data_slice, self.pyfai_config, precision_SAXS, segments=1, mask=self.mask)
        saxs_evo_results = saxs_evo_results.squeeze('frames')
        saxs_evo_results = saxs_evo_results.compute()
        self.saxs_mean_results = saxs_evo_results

    def calculate_saxs_evo_each_frame(self, precision_SAXS=600):
        raw_data_slice = self.slice_first_to_last_frame()
        saxs_evo_results = saxs.evolution_SAXS(raw_data_slice, self.pyfai_config, precision_SAXS, segments=len(raw_data_slice.frames), mask=self.mask)

        saxs_evo_results = saxs_evo_results.assign_coords(max_exp_time = ('frames', [(t+1)*saxs_evo_results.frames_per_segment*self.frame_time for t in range(saxs_evo_results['frames'].shape[0])])) #check if frame_time is enoug or if more complicated with dark times
        saxs_evo_results = saxs_evo_results.compute()
        self.saxs_evo_results_each_frame = saxs_evo_results

    def add_qrings(self, q_rings:QRings):
        """
        input: 
        """
        self.q_rings = q_rings

    def calculate_ttcfs(self, include_ttcf_via_std=False):
        """
        return: np.array(2D), np.array(3D), np.array(1D)
        Calculates the two-time correlation function for each partition provided by 'q_partitions' for the ScanSeries.
        """
        raw_data_slice = self.slice_first_to_last_frame()

        if include_ttcf_via_std == False:
            ttcfs = xpcs.calculate_ttcfs(raw_data_slice, self.q_rings, mask=self.mask.array)
        if include_ttcf_via_std == True:
            ttcfs, ttcfs_std = xpcs.calculate_ttcfs(raw_data_slice, self.q_rings, mask=self.mask.array,include_ttcf_via_std=True)

        ttcfs = ttcfs.assign_coords(t1 = ('frame1', [t*self.frame_time for t in range(ttcfs['frame1'].shape[0])]))#check if frame_time is enoug or if more complicated with dark times check if t2 is needed
        if include_ttcf_via_std == True:
            ttcfs_std = ttcfs_std.assign_coords(t1 = ('frame1', [t*self.frame_time for t in range(ttcfs['frame1'].shape[0])]))#check if frame_time is enoug or if more complicated with dark times check if t2 is needed

        if include_ttcf_via_std == False:
            ttcfs = ttcfs.compute()
            self.ttcf_data = ttcfs
        if include_ttcf_via_std == True:
            ttcfs, ttcfs_std = ttcfs.compute(), ttcfs_std.compute()
            self.ttcf_data = ttcfs
            self.ttcf_std_data = ttcfs_std

    #not yet reworked
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

#how to implement change this ?
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
 
