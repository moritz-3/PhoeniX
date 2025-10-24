# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import json

class Mask:
    """
    class Mask stores the mask used to indicate good (1) and bad (0) pixels.

    Attributes
    ----------
    filename : pathlib.Path
        Absolute path where the mask is saved as numpy array.
    array : np.array()
        Mask given as a numpy array where good (1) and bad (0) pixels are distinguished.
    array_boolean : np.array()
        Same as mask but converted to good (True) and bad (False).
    dim_y : int
        Number of pixels in y-direction
    dim_x : int
        Number of pixels in x-direction
    """
    def __init__(self, filename_mask):
        """
        input: Path - (path to the filename where the mask is saved) 
        Initialize an instance with all information obtained from a *.npy file describing good and bad pixels of the detector.
        """
        self.filename = filename_mask
        self.array = np.load(self.filename, allow_pickle=True)
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

def custom_encoder(obj):
    """
    return: str
    Converts input to string format if necessary. Ensures the content of the object can be written to a JSON file.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # add additional options if required
    return str(obj)

def process_incoherent_image(data3D, mask=None):
    """
    input: np.array(3D), _Mask 
    return: np.array(2D), np.array(2D)
    Calculates the incoherent image of a series of images.
    """
    if data3D.ndim != 3:
        raise ValueError("Dimensions of input array should be three (3D-array).")
    else:
        inc_img = np.mean(data3D, axis=0)
        dim_y, dim_x = inc_img.shape
        if mask is None: # only process the masked incoherent image in case a mask is provided.
            return (inc_img, None)
        if dim_y != mask.dim_y or dim_x != mask.dim_x:
            raise ValueError("Size of input data and mask do not match.")
        else:
            inc_img_cor = np.copy(inc_img)
            inc_img_cor[np.logical_not(mask.array_boolean)] = np.nan
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
    arrays : np.array()
        Partitions given as a np.arrays where included (1) and excluded (0) pixels are distinguished.
        Is a 3D array that has dimensions (#partitions, dim_y, dim_x)
    dim_y : int
        Number of pixels in y-direction (1st dimension of arrays)
    dim_x : int
        Number of pixels in x-direction (2nd dimension of arrays)
    info : dict
        Dictionary that contains additional information on how the q-partitions have been created (e.g., as Q-rings if similar Q-value)
    """
    def __init__(self):
        self.arrays = np.array(None)
        self.dim_y = 0
        self.dim_x = 0
        self.info = {}

    def q_partitions_from_q_rings(self, q_centers, q_widths, detector_shape, poni_info):
        """
        input: np.array, np.array, (int, int), pyFAI-poni
        Generates matrices with the shape of the detector that define which pixels are included in a given ring. 
        Also updates the attribute 'info', a dictionary that contains the information on how these Q-rings have been determined.
        Check 'help(QRings)' for further information.
        """
        self.dim_y, self.dim_x = detector_shape
        q_map = poni_info.qArray(detector_shape) # can be used to get the q-value for each pixel. The output is given in units of 1/nm.
        q_map_inversem = np.array(q_map)*1e9
        q_mins = q_centers - q_widths/2
        q_maxs = q_centers + q_widths/2
        number_partitions = len(q_centers)
        partitions = np.zeros((number_partitions, self.dim_y, self.dim_x))
        for i in range(number_partitions):
            partitions[i,:,:] = np.logical_and(q_map_inversem > q_mins[i], q_map_inversem < q_maxs[i])
        self.arrays = partitions
        self.info['description'] = 'Q-partitions defined as Q-rings'
        self.info['number_partitions'] = number_partitions
        self.info['q_centers'] = q_centers
        self.info['q_widths'] = q_widths



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