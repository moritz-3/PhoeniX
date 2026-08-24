# -*- coding: utf-8 -*-
"""
to do:  add units where needed
        mave incoherant image function to a more suitable place
        Unify Q-rings and Q-Partitions if possible. Should be usable in transmission and reflection setups
"""
import numpy as np
import json
import xarray as xr
import matplotlib

class Mask:
    """
    class Mask stores the mask used to indicate good (1) and bad (0) pixels.

    Attributes
    ----------
    filename : pathlib.Path
        Absolute path where the mask is saved as numpy array.

    array : xr.DataArray
        Mask given as a xarray data array with dimensions x and y and name meask where good (1) and bad (0) pixels are distinguished.

    array_boolean : xr.DataArray
        Same as mask but converted to good (True) and bad (False).

    dim_y : int
        Number of pixels in y-direction

    dim_x : int
        Number of pixels in x-direction

    To Do:
    ____________
    add plot function.scan_series.ttcf_data
    """
    def __init__(self, filename_mask):
        """
        input: Path - (path to the filename where the mask is saved) 
        Initialize an instance with all information obtained from a *.npy file describing good and bad pixels of the detector.
        """
        self.filename = filename_mask
        self.array = np.load(self.filename, allow_pickle=True)
        self.array = xr.DataArray(self.array, dims=['y','x'], name='Mask')
        self.array_boolean = self.array.astype(bool)
        self.dim_y, self.dim_x = self.array.shape

    def create_dict_output(self):
        """
        return: dict
        Creates a single dictionary as output compatible with JSON files.
        """
        all_parameters = {'Filename': str(self.filename), 'Dimensions (y,x)': (self.dim_y, self.dim_x)}
        return all_parameters

    def __str__(self):
        """
        return: str
        Returns the object as a JSON file.
        """
        return json.dumps(self.create_dict_output(), indent=4, default=custom_encoder)

    def show_mask(self):
        cmap = matplotlib.colors.ListedColormap(['tab:blue', 'yellow'])
        bounds = [-0.5, 0.5, 1.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        im = ax.imshow(self.array_boolean, origin='lower', cmap=cmap, norm=norm)
        cbar = plt.colorbar(im, ticks=[0,1])
        cbar.set_ticklabels(['Bad', 'Good'])
        ax.set_xlabel(r'Pixel$_x$')
        ax.set_ylabel(r'Pixel$_y$')
        ax.set_title('Mask')
        plt.show()

def custom_encoder(obj):
    """
    return: str
    Converts input to string format if necessary. Ensures the content of the object can be written to a JSON file.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # add additional options if required
    return str(obj)

def process_incoherent_image(data3D:xr.DataArray, mask=None):
    """
    Can we get rid of this?
    calculates the mean of the raw data along the frames axis without the mask and if a mask if given additionaly with mask

    Parameters:
    ___________
    data3D : xr.DataArray
        scans.ScanSeries.raw_data

    mask : Mask
        Mask object that contains xr.DataArray with mask data and other mask related parameters

    Returns:
    __________
    xr.DataArray
        mean raw data along the frames axis

    xr.DataArray
        if a mask if given returns the mean raw data along frames axis with masked pixels excluded

    Raises Error:
    ___________
    If input data3D is not three dimensional

    If mask and raw data x and y axis do not match
    """

    if data3D.ndim != 3:
        raise ValueError("Dimensions of input array should be three (3D-array).")

    else:
        inc_img = data3D.mean(dim='frames', skipna=True)
        dim_y, dim_x = inc_img.shape
        inc_img = inc_img.rename('mean_raw_data')

        if mask is None: # only process the masked incoherent image in case a mask is provided.
            return (inc_img, None)

        if dim_y != mask.dim_y or dim_x != mask.dim_x:
            raise ValueError("Size of input data and mask do not match.")

        else:
            inc_img_cor = inc_img
            inc_img_cor = inc_img_cor.where(mask.array!=0, np.nan)
            inc_img_cor = inc_img_cor.rename('mask_mean_data')
            return (inc_img, inc_img_cor)

class QPartitions:
    """
    class QPartitions stores arrays used to define pixel selections. It needs to match in dimensions with mask and data.
    Within each partition, included pixels (1) can be distinguished from excluded pixels (0).

    Attributes
    ----------
    filename : pathlib.Path
        Absolute path where the Q-partitions are saved as np.array.
    arrays : np.array()
        Partitions given as a np.arrays where included (1) and excluded (0) pixels are distinguished.
        Is a 3D array that has dimensions (#partitions, dim_y, dim_x)
    number_q_partitions : int
        Number of partitions (0th dimension of arrays).
    dim_y : int
        Number of pixels in y-direction (1st dimension of arrays)
    dim_x : int
        Number of pixels in x-direction (2nd dimension of arrays)
    info : dict
        Dictionary that contains additional information on how the q-partitions have been created (e.g., as Q-rings if similar Q-value)
    """
    def __init__(self):
        """
        Initialize an instance with all attributes defined by dummy values.
        """
        self.filename = 'no file'
        self.arrays = np.array(None)
        self.number_q_partitions = 0
        self.dim_y = 0
        self.dim_x = 0
        self.info = {}

    def define_partitions(self, partition_arrays, info={}):
        """
        input: np.array(3D), _dict - [q-partitions (dim_#partitions, dim_y, dim_x), info]
        Defines the Q-partitions to the instance and updates all attributes.
        Note that if the instance already contains q-partitions or info, these values become overwritten. 
        In case you would like to add partitions to an already existing selection of partitions, use 'add_partitions' instead.
        """
        self.info = info
        self.arrays = np.array(partition_arrays)
        self.number_q_partitions, self.dim_y, self.dim_x = self.arrays.shape

    def add_partitions(self, partition_arrays, info={}):
        """
        """
        # ensure that the existing partitions and new partitions match in dimensions
        add_arrays = np.array(partition_arrays)
        add_number_partitions, add_dim_y, add_dim_x = add_arrays.shape
        if add_dim_y != self.dim_y or add_dim_x != self.dim_x:
            raise ValueError("Dimensions of existing and new partitions do not agree. Consider using 'define_partitions' instead.")
        else:
            # update info (creates a new entry with info on the additional partitions)
            base_key = "additional_partitions"
            key = base_key
            counter = 1
            while key in self.info:
                key = f'{base_key}_{counter}'
                counter += 1
            self.info[key] = info
            # update arrays and #partitions
            self.arrays = np.concatenate((self.arrays, add_arrays), axis=0)
            self.number_q_partitions = self.arrays.shape[0]

    def save_partitions(self, filename_save):
        """
        """
        self.filename = filename_save
        np.save(self.filename, self.arrays)

    def create_dict_output(self):
        """
        return: dict
        Creates a single dictionary as output compatible with JSON files.
        """
        all_parameters = {'Filename': str(self.filename), 'Dimensions (y,x)': (self.dim_y, self.dim_x), '#Partitions': self.number_q_partitions, 'Info': self.info}
        return all_parameters

class QRings:
    """
    class QRings is intended to create Q-partitions from information given by a poni file and detector dimensions.
    It can be used as an input for creating an instance of QPartitions.
    Within each partition, included pixels (1) can be distinguished from excluded pixels (0).
    It is especially suitable for SAXS-XPCS and USAXS-XPCS measurements.
    In the future it will also applicable to WAXS-XPCS measurements.

    Attributes
    ----------
    mask_arrays : xr.DataArray
        Partitions where included (1) and excluded (0) pixels are distinguished.
        Is a 3D array that has dimensions (#partitions, dim_y, dim_x)
    dim_y : int
        Number of pixels in y-direction (1st dimension of arrays)
    dim_x : int
        Number of pixels in x-direction (2nd dimension of arrays)
    info : dict
        Dictionary that contains additional information on how the q-partitions have been created (e.g., as Q-rings if similar Q-value)

    To Do:
    _________
    Discuss info implementation
    """
    def __init__(self, q_centers, q_widths, detector_shape, poni_info):
        """
        Parameters:
        ___________
        q_centers : np.array
                    array with centers of the q ring as radius in [unit] from the beamcenter

        q_widths : np.array
                    array with width of each of the q-rings. Should have the same dimension as q_centers

        detector_shape : tuple(int,int)
                    schape of the detector in (dim y, dim x)

        poni_info : 
                pyFAI object with information about the q-space of the detector

        """
        self.dim_y = detector_shape[0]
        self.dim_x = detector_shape[1]
        self.q_centers = q_centers
        self.q_widths = q_widths
        #self.info = {}
        if len(q_centers) != len(q_widths): raise ValueError('q_centers and q_widths must be of the same lenght')
        q_map = poni_info.qArray(detector_shape) # can be used to get the q-value for each pixel. The output is given in units of 1/nm.
        q_map_inverse_m = np.array(q_map)*1e9
        q_mins = q_centers - q_widths/2
        q_maxs = q_centers + q_widths/2
        number_partitions = len(q_centers)
        partitions = np.zeros((number_partitions, self.dim_y, self.dim_x))
        for i in range(number_partitions):
            partitions[i] = np.logical_and(q_map_inverse_m > q_mins[i], q_map_inverse_m < q_maxs[i])
        qring_masks = xr.DataArray(data = partitions,
                                  dims = ('partitions','y','x'),
                                  name = 'q-ring masks',
                                  )
        self.mask_arrays = qring_masks

        """
        self.arrays = partitions
        self.info['description'] = 'Q-partitions defined as Q-rings'
        self.info['number_partitions'] = number_partitions
        self.info['q_centers'] = q_centers
        self.info['q_widths'] = q_widths
        """


    # def show_q_rings(self, SAXS_Q, SAXS_IofQ, y_limits=None):
    #     fig, ax = plt.subplots()
    #     ax.plot(SAXS_Q*1e-9, SAXS_IofQ)
    #     if y_limits is None:
    #         y_limits = (0.9*np.nanmin(SAXS_IofQ), 1.1*np.nanmax(SAXS_IofQ))
    #     ax.set_yscale('log')
    #     ax.set_ylim(y_limits)
    #     q_centers = self.info['q_centers']
    #     q_widths = self.info['q_widths']
    #     q_mins = q_centers - q_widths/2
    #     num_rings = len(q_widths)
    #     # define colors for ymap
    #     cmap = plt.get_cmap('jet')
    #     colors = cmap(np.linspace(0, 1, num_rings))
    #     for i in range(num_rings):
    #         rectangle = patches.Rectangle((q_mins[i]*1e-9, y_limits[0]), q_widths[i]*1e-9, y_limits[1] - y_limits[0], alpha=0.4, edgecolor='black', facecolor=colors[i], linewidth=1)
    #         ax.add_patch(rectangle)
    #     plt.show()
