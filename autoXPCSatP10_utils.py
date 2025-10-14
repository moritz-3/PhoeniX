import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # to draw rectangles and circles for q-bins
import matplotlib.cm as cm # to create a colormap to be assigned to the q-bins
from matplotlib.colors import LogNorm
import scipy
import time
import datetime as dt
import h5py
import hdf5plugin
from pathlib import Path
import re # for regular expressions
import pyFAI
import pyFAI.azimuthalIntegrator
# import psutil (only in jupyter notebook?)
# import gc # garbage collector (only in jupyter notebook?)

### SAMPLE ###
class Sample:
    def __init__(self, directory_beamtime, sample_name):
        self.directory_beamtime = Path(directory_beamtime)
        self.sample_name = sample_name

### MASK ###
class Mask:
    def __init__(self, directory_mask, filename_mask):
        self.directory_mask = Path(directory_mask)
        self.filename_mask = filename_mask
        self.mask = np.load(self.directory_mask / self.filename_mask)
        self.mask_boolean = self.mask.astype(bool)
        self.dim_y_mask, self.dim_x_mask = self.mask.shape

### EXPERIMENTAL PARAMETERS ###
class Experimental_Parameters:
    def __init__(self):
        self.all_attributes_defined = False
        self.center_x = None
        self.center_y = None
        self.energy_eV = 0
        self.wavelength_m = 0
        self.number_frames_raw = 0
        self.pixel_size_m = 0
        self.sample_detector_distance_m = 0
        self.time_interval_s = 0

    def update_status_all_attributes_defined(self):
        self.all_attributes_defined = not ((self.center_x is None) 
                                           or (self.center_y is None)
                                           or (self.energy_eV is None)
                                           or (self.wavelength_m is None)
                                           or (self.number_frames_raw is None)
                                           or (self.pixel_size_m is None)
                                           or (self.sample_detector_distance_m is None)
                                           or (self.time_interval_s is None))

    def extract_detector_pixel_size(self, scan):
        base_path = scan.sample.directory_beamtime / f'raw/{scan.sample.sample_name}_{scan.scan_number:05d}'
        filename_fio = f'{scan.sample.sample_name}_{scan.scan_number:05d}.fio'
        with open(base_path / filename_fio, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split()
                if line[0] == '_pixelsize':
                    self.pixel_size_m = float(line[2]) * 1e-3
        self.update_status_all_attributes_defined()

    def extract_time_interval(self, scan):
        base_path = scan.sample.directory_beamtime / f'raw/{scan.sample.sample_name}_{scan.scan_number:05d}/{scan.detector}'
        filename_master = f'{scan.sample.sample_name}_{scan.scan_number:05d}_master.h5'
        with h5py.File(base_path / filename_master, 'r') as file:
            time_interval = file['entry/instrument/detector/frame_time']
            self.time_interval_s = time_interval[()]
        self.update_status_all_attributes_defined()

    def extract_general_parameters(self, scan, pixel_size):
        base_path = scan.sample.directory_beamtime / f'raw/{scan.sample.sample_name}_{scan.scan_number:05d}/{scan.detector}'
        filename_batchinfo = f'{scan.sample.sample_name}_{scan.scan_number:05d}.batchinfo'
        with open(base_path / filename_batchinfo, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split()
                match line[0]:
                    case 'energy:':
                        self.energy_eV = float(line[1])*1e3 
                    case 'rr:':
                        self.sample_detector_distance_m = float(line[1])*1e-3
                    case 'ndataend:':
                        self.number_frames_raw = int(line[1][1:-1])
                    case 'x0:':
                        x_0 = int(line[1]) 
                    case 'y0:':
                        y_0 = int(line[1])
                    case 'ccdx:':
                        ccdx = float(line[1])*1e-3
                    case 'ccdz:':
                        ccdz = float(line[1])*1e-3
                    case 'ccdx0:':
                        ccdx0 = float(line[1])*1e-3
                    case 'ccdz0:':
                        ccdz0 = float(line[1])*1e-3
        self.center_x = x_0 + (ccdx - ccdx0) / pixel_size
        self.center_y = y_0 + (ccdz - ccdz0) / pixel_size
        self.wavelength_m = scipy.constants.h * scipy.constants.speed_of_light / (self.energy_eV * scipy.constants.e)
        self.update_status_all_attributes_defined()

    def update_all_experimental_parameters(self, scan):
        self.extract_detector_pixel_size(scan)
        self.extract_time_interval(scan)
        self.extract_general_parameters(scan, self.pixel_size_m)

####### Q-Partitions ######### 
class Q_Partitions:
    def __init__(self):
        self.number_q_partitions = 0
        self.q_partitions = None
        self.colors_q_partitions = None
        self.dim_x_q_partitions = None
        self.dim_y_q_partitions = None

    def define_colors_q_partitions(self):
        if self.number_q_partitions >= 1:
            self.colors_q_partitions = [cm.get_cmap('jet')(i / (self.number_q_partitions - 1)) for i in range(self.number_q_partitions)]

    def create_q_partitions_from_q_rings(self, q_rings, scan):
        self.q_rings = q_rings
        self.number_q_partitions = q_rings.number_q_rings
        self.q_partitions = np.full((q_rings.number_q_rings, scan.dim_y, scan.dim_x), False, dtype=bool)
        for i in range(q_rings.number_q_rings):
            radius_inner = scan.experimental_parameters.sample_detector_distance_m / scan.experimental_parameters.pixel_size_m * np.tan(2*np.arcsin((q_rings.q_radii[i] - q_rings.q_widths[i]/2) * scan.experimental_parameters.wavelength_m / (4*scipy.constants.pi)))
            radius_outer = scan.experimental_parameters.sample_detector_distance_m / scan.experimental_parameters.pixel_size_m * np.tan(2*np.arcsin((q_rings.q_radii[i] + q_rings.q_widths[i]/2) * scan.experimental_parameters.wavelength_m / (4*scipy.constants.pi)))
            Y, X = np.ogrid[:scan.dim_y, :scan.dim_x]
            distance_squared = (X - scan.experimental_parameters.center_x)**2 + (Y - scan.experimental_parameters.center_y)**2
            next_q_partition = (distance_squared < radius_outer**2) & (distance_squared > radius_inner**2)
            self.q_partitions[i,next_q_partition] = True

    # Show Q-ring partitions
    def show_q_ring_partitions(self, scan, results_SAXS, show_figure, vmax):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(scan.incoherent_img, vmin=0, vmax=vmax, origin='lower')
        for i in range(self.number_q_partitions):
            next_radius = scan.experimental_parameters.sample_detector_distance_m / scan.experimental_parameters.pixel_size_m * np.tan(2*np.arcsin((self.q_rings.q_radii[i] - self.q_rings.q_widths[i]/2) * scan.experimental_parameters.wavelength_m / (4*scipy.constants.pi)))
            tmp = scan.experimental_parameters.sample_detector_distance_m / scan.experimental_parameters.pixel_size_m * np.tan(2*np.arcsin((self.q_rings.q_radii[i] + self.q_rings.q_widths[i]/2) * scan.experimental_parameters.wavelength_m / (4*scipy.constants.pi)))
            next_width = tmp - next_radius
            ring = patches.Wedge(center=(scan.experimental_parameters.center_x,scan.experimental_parameters.center_y), r=next_radius, theta1=0, theta2=360, width = next_width, alpha=0.4, edgecolor='black', facecolor=self.colors_q_partitions[i], linewidth=1)
            axes[0].add_patch(ring)
        axes[0].set_title("2D image")

        axes[1].plot(results_SAXS.SAXS_Q*1e-9,results_SAXS.SAXS_IofQ)
        ymin_plot, ymax_plot = axes[1].get_ylim()
        axes[1].set_title('SAXS curve')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Q (nm-1)')
        axes[1].set_ylabel('Intensity (arb. u.)')
        for i in range(self.number_q_partitions):
            rectangle = patches.Rectangle(((self.q_rings.q_radii[i] - self.q_rings.q_widths[i]/2)*1e-9, ymin_plot), self.q_rings.q_widths[i]*1e-9, ymax_plot - ymin_plot, alpha=0.4, edgecolor='black', facecolor=self.colors_q_partitions[i], linewidth=1)
            axes[1].add_patch(rectangle)

        plt.tight_layout()
        if show_figure:
            plt.show()
        return fig

### Q_RINGS ###
class Q_Rings:
    def __init__(self, q_radii, q_widths):
        self.q_radii = q_radii
        self.q_widths = q_widths
        self.number_q_rings = len(q_radii)
        

### SCAN ###
class Scan:
    def __init__(self, sample, detector, scan_number, skip_frames_start, experimental_parameters):
        self.sample = sample
        self.detector = detector
        self.raw_data = None
        self.dim_x = None
        self.dim_y = None
        self.scan_number = scan_number
        self.skip_frames_start = skip_frames_start
        self.experimental_parameters = experimental_parameters
        self.mask = None
        self.incoherent_img = None
        self.start_time = None
        self.end_time = None

    def provide_mask(self, mask):
        self.mask = mask

    # check if raw data exists for a scan
    def exists_raw_data(self):
        base_path = self.sample.directory_beamtime / f'raw/{self.sample.sample_name}_{self.scan_number:05d}/{self.detector}'
        return base_path.exists()

    def load_raw_data(self):
        base_path = self.sample.directory_beamtime / f'raw/{self.sample.sample_name}_{self.scan_number:05d}/{self.detector}'
        filename_selection = list(base_path.glob(f'{self.sample.sample_name}_{self.scan_number:05d}_data_*.h5'))
        filename_selection.sort(key=lambda f: int(re.search(r'_data_(\d{6})\.h5$', f.name).group(1)))
        with h5py.File(filename_selection[0], 'r') as f:
            _, self.dim_y, self.dim_x = f['entry/data/data'].shape
        dimensions = (self.experimental_parameters.number_frames_raw, self.dim_y, self.dim_x)
        all_data = np.zeros(dimensions, dtype='float32')
        start_frame = 0
        for filename in filename_selection:
            with h5py.File(filename) as f:
                data = f['entry']['data']['data']
                next_data = data[:]
                next_num_frames = next_data.shape[0]
                all_data[start_frame:start_frame + next_num_frames,:,:] = next_data
                start_frame += next_num_frames
        self.raw_data = all_data[self.skip_frames_start:,:,:]

    def process_incoherent_image(self, show_figure, plot_max):
        self.incoherent_img = np.mean(self.raw_data, axis=0)
        if self.mask is None or not (self.dim_x == self.mask.dim_x_mask and self.dim_y == self.mask.dim_y_mask):
            if show_figure:
                plt.figure()
                plt.imshow(self.incoherent_img, origin='lower', vmin=0, vmax=plot_max)
                plt.xlabel('Pixel_x')
                plt.ylabel('Pixel_y')
            #return the incoherent image and None value
            return self.incoherent_img, None
        else:
            incoherent_img_cor = np.copy(self.incoherent_img)
            incoherent_img_cor[np.logical_not(self.mask.mask_boolean)] = np.nan
            # show image
            if show_figure:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(self.incoherent_img, origin='lower', vmin=0, vmax=plot_max)
                axes[0].set_title("Inc. Image")
    
                axes[1].imshow(self.mask.mask_boolean, origin='lower')
                axes[1].set_title('Mask')
    
                axes[2].imshow(incoherent_img_cor, origin='lower', vmin=0, vmax=plot_max)
                axes[2].set_title('Inc. Image Corrected')
                plt.tight_layout()
                plt.show()
            # return the incoherent and corrected incoherent image
            return self.incoherent_img, incoherent_img_cor

    # Calculate SAXS curve and return an instance of Results_SAXS
    def process_SAXS(self, precision_SAXS):
        # create pyFAI configuration (corresponds to a poni file)
        pyfai_configuration = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
            dist = self.experimental_parameters.sample_detector_distance_m, # distance (m)
            pixel1 = self.experimental_parameters.pixel_size_m, # pixel size (m) in y-direction
            pixel2 = self.experimental_parameters.pixel_size_m, # pixel size (m) in x-direction
            poni1 = self.experimental_parameters.center_y * self.experimental_parameters.pixel_size_m, # position of direct beam (m) in y-direction
            poni2 = self.experimental_parameters.center_x * self.experimental_parameters.pixel_size_m, # position of direct beam (m) in x-direction
            rot1 = 0, # assuming the detector is perpendicular to the direct beam
            wavelength = self.experimental_parameters.wavelength_m # wavelength (m)
        )
        mask = self.mask
        if (mask is None or self.dim_x != mask.dim_x_mask or self.dim_y != mask.dim_y_mask):
            mask_array = None
        else:
            mask_array = np.logical_not(mask.mask_boolean) # pyFAI requires the inverted mask
        # calculate SAXS curve
        SAXS_Q, SAXS_IofQ, SAXS_errorIofQ = pyfai_configuration.integrate1d(data = self.incoherent_img, # 2D data array
            npt = precision_SAXS,
            unit='q_nm^-1', # unit of x-axis
            mask=mask_array,
            error_model = 'poisson')
        # create Results_SAXS object
        results_SAXS = Results_SAXS(precision_SAXS, SAXS_Q*1e9, SAXS_IofQ, SAXS_errorIofQ, pyfai_configuration)
        return results_SAXS

    def process_SAXS_slices(self, average_over_frames, precision_SAXS): # not included in automatic save yet
        # create pyFAI configuration (corresponds to a poni file)
        pyfai_configuration = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
            dist = self.experimental_parameters.sample_detector_distance_m, # distance (m)
            pixel1 = self.experimental_parameters.pixel_size_m, # pixel size (m) in y-direction
            pixel2 = self.experimental_parameters.pixel_size_m, # pixel size (m) in x-direction
            poni1 = self.experimental_parameters.center_y * self.experimental_parameters.pixel_size_m, # position of direct beam (m) in y-direction
            poni2 = self.experimental_parameters.center_x * self.experimental_parameters.pixel_size_m, # position of direct beam (m) in x-direction
            rot1 = 0, # assuming the detector is perpendicular to the direct beam
            wavelength = self.experimental_parameters.wavelength_m # wavelength (m)
        )
        mask = self.mask
        if (mask is None or self.dim_x != mask.dim_x_mask or self.dim_y != mask.dim_y_mask):
            mask_array = None
        else:
            mask_array = np.logical_not(mask.mask_boolean) # pyFAI requires the inverted mask
        # calculate SAXS curves
        i=0
        SAXS_IofQ_list = []
        SAXS_errorIofQ_list = []
        while i <= len(self.raw_data):
            j = i + average_over_frames
            if j > len(self.raw_data): # makes the last slices smaller than the rest but makes sure all data is included
                j=len(self.raw_data)
            data_slice = np.mean(self.raw_data[i:j], axis=0)
            
            SAXS_Q, SAXS_IofQ, SAXS_errorIofQ = pyfai_configuration.integrate1d(data = data_slice, # 2D data array
                        npt = precision_SAXS,
                        unit='q_nm^-1', # unit of x-axis
                        mask=mask_array,
                        error_model = 'poisson')

            SAXS_IofQ_list.append(SAXS_IofQ)
            SAXS_errorIofQ_list.append(SAXS_errorIofQ)
            
            i += average_over_frames

        SAXS_IofQ_list = np.array(SAXS_IofQ_list)
        SAXS_errorIofQ_list = np.array(SAXS_errorIofQ_list)
        
       # create Results_SAXS_slices object
        results_SAXS_slices = Results_SAXS_slices(average_over_frames, precision_SAXS, SAXS_Q*1e9, SAXS_IofQ_list, SAXS_errorIofQ_list,  pyfai_configuration)
        return results_SAXS_slices

    def calculate_ttcs(self, q_partitions):
        number_frames_analyzed = self.experimental_parameters.number_frames_raw - self.skip_frames_start
        intensities_per_q_partition = np.zeros((q_partitions.number_q_partitions, number_frames_analyzed))
        pixels_per_q_partition = np.zeros(q_partitions.number_q_partitions)
        ttc_per_q_partition = np.zeros((q_partitions.number_q_partitions, number_frames_analyzed, number_frames_analyzed))
        for i in range(q_partitions.number_q_partitions):
            # combine general mask and q-partition
            next_mask = np.logical_and(q_partitions.q_partitions[i,:,:],self.mask.mask_boolean)
            # reduce dataset from 3D (number_frames, dim_y, dim_x) to 2D (number_frames, dim_xy) with dim_xy determined by the mask
            next_data_2D = self.raw_data[:,next_mask]
            # calculate ttc
            next_intensity_per_q_partition, next_number_of_pixels, next_ttc = calculate_ttc(next_data_2D)
            intensities_per_q_partition[i,:] = next_intensity_per_q_partition
            pixels_per_q_partition[i] = next_number_of_pixels
            ttc_per_q_partition[i,:,:] = next_ttc

        results_TTC = Results_TTC(intensity_per_q_partition=intensities_per_q_partition, ttc_per_q_partition=ttc_per_q_partition, pixels_per_q_partition=pixels_per_q_partition, q_partitions=q_partitions)
        return results_TTC

    def get_start_end_time(self): #could be improved with datetime
        path = self.sample.directory_beamtime / f'raw/{self.sample.sample_name}_{self.scan_number:05d}/{self.detector}/{self.sample.sample_name}_{self.scan_number:05d}.batchinfo'
        with open(path, 'r') as f:
            for ln in f:
                if 'start_time' in ln:
                    start_time = ln[:-2]
                if 'end_time' in ln:
                    end_time = ln[:-2]
        self.start_time = start_time.split()[2:]
        self.end_time = end_time.split()[2:]

    def update_end_time(self): # only use this if start and end time in the batchinfo file are the same
        self.get_start_end_time()
        exp_time = self.experimental_parameters.number_frames_raw * self.experimental_parameters.time_interval_s
        dt_start_time, dt_end_time = self.time_to_datetime() 
        end_time = dt_start_time + dt.timedelta(seconds=exp_time)
        self.end_time = [end_time.strftime("%b"), end_time.strftime("%d"), end_time.strftime("%H:%M:%S"), end_time.strftime("%Y")]

    def time_to_datetime(self):
        start_time_str = self.start_time[0] + self.start_time[1] + self.start_time[2] + self.start_time[3]
        end_time_str = self.end_time[0] + self.end_time[1] + self.end_time[2] + self.end_time[3]
        dt_start_time = dt.datetime.strptime(start_time_str, "%b%d%H:%M:%S%Y")
        dt_end_time = dt.datetime.strptime(end_time_str, "%b%d%H:%M:%S%Y")
        return dt_start_time, dt_end_time
        

    def start_end_time_to_str(self):
        start_time = f'{self.start_time[3]}'+month_three_letters_to_number(self.start_time[0])+f'{self.start_time[1]}{self.start_time[2]}'.replace(':','')
        end_time = f'{self.end_time[3]}'+ month_three_letters_to_number(self.end_time[0])+f'{self.end_time[1]}{self.end_time[2]}'.replace(':','')  
        return start_time, end_time

    def get_temp_file_data(self, path_to_temp_file): # skip frames no yet taken into account
        f = np.genfromtxt(path_to_temp_file, skip_header=True)   
        start_time, end_time = self.start_end_time_to_str()
        for i in range(len(f)):
            if  start_time == str(int(f[i][0])) or start_time == str(int(f[i][0])+1): # or with +1 easy fix band-aid sulution because temp file only includes data evry 2s
                start_id = i
            if end_time == str(int(f[i][0])) or end_time == str(int(f[i][0])+1):
                end_id = i+1
                break
                
        temp_time_list= f[start_id:end_id,0]
        temp_time_list = [dt.datetime.strptime(str(int(x)),"%Y%m%d%H%M%S") for x in temp_time_list]
        temp_time_list = [(x - temp_time_list[0]).total_seconds() for x in temp_time_list]
        temp_set_point_list = f[start_id:end_id,1]
        temp_sensorA_list = f[start_id:end_id,2]
        temp_sensorB_list = f[start_id:end_id,3]
        temp_heater_output_list = f[start_id:end_id,4]
        temp_range_list =  f[start_id:end_id,5]

        temp_data = Results_temp(temp_time_list, temp_set_point_list, temp_sensorA_list, temp_sensorB_list, temp_heater_output_list, temp_range_list)
        return temp_data
       

### PROCESSING RESULTS ###
class Processing_Results:
    def __init__(self, scan, directory_general_saving, overwrite=False):
        self.scan = scan
        if overwrite:
            self.directory_saving = directory_general_saving / f'{self.scan.sample.sample_name}_{self.scan.scan_number:05d}'
        else:
            self.directory_saving = self.create_new_output_directory(directory_general_saving)
        print(f'Results will be saved under:\n{self.directory_saving}.')
        self.results_incoherent_img = None
        self.results_incoherent_img_cor = None
        self.results_SAXS = None
        self.results_saxs_slices = None
        self.results_TTC = None
        self.q_partitions_figure = None
        self.results_temp_data = None

    def create_new_output_directory(self, directory_general_saving):
        base_path = directory_general_saving / f'{self.scan.sample.sample_name}_{self.scan.scan_number:05d}'
        if not base_path.exists():
            base_path.mkdir(parents=True)
            directory_saving = base_path
        else:
            counter = 1
            while True:
                new_path = base_path.parent / f'{base_path.name}_{counter:02d}'
                if not new_path.exists():
                    new_path.mkdir(parents=True)
                    directory_saving = new_path
                    break
                else:
                    counter += 1

        return directory_saving

    def add_saxs_results(self, results_SAXS):
        self.results_SAXS = results_SAXS

    def add_ttc_results(self, results_TTC):
        self.results_TTC = results_TTC

    def add_saxs_slices_results(self, results_saxs_slices):
        self.results_saxs_slices = results_saxs_slices
        
    def add_temp_data(self, results_temp_data):
        self.results_temp_data = results_temp_data
    
    def generate_output(self, saxs_slices=False, temp_data=False): 
        np.save(self.directory_saving / 'q_partitions', self.results_TTC.q_partitions.q_partitions)
        self.q_partitions_figure.savefig(self.directory_saving / 'q_partitions_visualization.png', dpi=300, bbox_inches='tight')
        np.save(self.directory_saving / 'incoherent_img', self.results_incoherent_img)
        np.save(self.directory_saving / 'incoherent_img_cor', self.results_incoherent_img_cor)
        SAXS_data_for_saving = np.stack((self.results_SAXS.SAXS_Q, self.results_SAXS.SAXS_IofQ, self.results_SAXS.SAXS_errorIofQ), axis=1)
        np.save(self.directory_saving / 'SAXS_data', SAXS_data_for_saving)
        self.results_SAXS.pyFAIconfig.save('pyfai_config_SAXS.poni')
        np.save(self.directory_saving / 'intensity_variation_per_q_partition', self.results_TTC.intensity_per_q_partition)
        np.save(self.directory_saving / 'all_ttc', self.results_TTC.ttc_per_q_partition) 
        np.save(self.directory_saving / 'all_g2', self.results_TTC.g2_functions)
        if saxs_slices==True:
            np.save(self.directory_saving / 'SAXS_data_slices', self.results_saxs_slices)
        if temp_data==True:
            np.save(self.directory_saving / 'temp_data', self.results_temp_data)

    def write_output_summary(self):
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.directory_saving / 'overview_input_output.txt', "w") as f:
            f.write("#Input:\n")
            f.write(f'Date and Time: {time_stamp}\n')
            f.write(f'Beamtime: {self.scan.sample.directory_beamtime}\n')
            f.write(f'Sample: {self.scan.sample.sample_name}\n')
            f.write(f'Scan: {self.scan.scan_number:05d}\n')
            f.write(f'Detector: {self.scan.detector}\n')
            f.write(f'Detector Dimensions (x,y): ({self.scan.dim_x},{self.scan.dim_y})\n')
            f.write(f'Number of Frames: {self.scan.experimental_parameters.number_frames_raw}\n')
            f.write(f'Number of Initial Frames Skipped for Data Analysis: {self.scan.skip_frames_start}\n')
            f.write(f'Beam Center (x,y) in pixels: ({self.scan.experimental_parameters.center_x},{self.scan.experimental_parameters.center_y})\n')
            f.write(f'Energy (eV): {self.scan.experimental_parameters.energy_eV}\n')
            f.write(f'Wavelength (m): {self.scan.experimental_parameters.wavelength_m}\n')
            f.write(f'Sample to Detector Distance: {self.scan.experimental_parameters.sample_detector_distance_m}\n')
            f.write(f'Time Interval/Delay Time (Time Between Frames): {self.scan.experimental_parameters.time_interval_s}\n')
            f.write('\n')
            # mask
            f.write(f'Mask directory: {self.scan.mask.directory_mask}\n')
            f.write(f'Mask filename: {self.scan.mask.filename_mask}\n')
            # q-partitions
            f.write(f'Q-Ring Centers: {self.results_TTC.q_partitions.q_rings.q_radii}\n')
            f.write(f'Q-Ring Widths: {self.results_TTC.q_partitions.q_rings.q_widths}\n')
            f.write('\n')

            f.write('#Output:\n')
            # q-partitions
            f.write('Q-Partitions can be found under: q_partitions.npy')
            f.write('Dimensions: (Number of Q-Partitions, Dimension y, Dimension x)\n')
            f.write('A visualization of the Q-Paritions on the Detector and on the SAXS curve can be found under: q_partitions_visualization.png\n')
            f.write('Note that the Q-Partitions have the size of the detector. In combination with the mask, they provide the pixels considered per q-bin.\n')
            # incoherent images
            f.write('The detector image summed over all frames (typically corresponding to the incoherent image) can be found under: incoherent_img.npy\n')
            f.write('The same image but with a correction by bad pixels as provided by the mask can be found under: incoherent_img_cor.npy\n')
            # SAXS curve
            f.write('SAXS_data.npy:\n')
            f.write('- Contains the mean SAXS-curve of the scan series.\n')
            f.write(f'- Shape: ({self.results_SAXS.precision_SAXS}, 3) with [q-values in m, intensities, error in intensities]\n')
            # pyFAI poni file
            f.write('pyfai_config_SAXS.poni:\n')
            f.write('- *.poni file for processing via pyFAI module for area detectors.\n')
            f.write('intensity_variation_per_q_partition.npy\n')
            f.write('- Contains the change in intensity for every q-partition per frame.\n')
            f.write(f'- Shape (number of q-partitions, number of frames analyzed): ({self.results_TTC.q_partitions.number_q_partitions},{self.scan.experimental_parameters.number_frames_raw - self.scan.skip_frames_start})\n')
            f.write('all_ttc.npy:\n')
            f.write('- Contains TTCs for the q-partitions specified.\n')
            f.write(f'- Shape (number of q-partitions, number of frames analyzed, number of frames analyzed): ({self.results_TTC.q_partitions.number_q_partitions},{self.scan.experimental_parameters.number_frames_raw - self.scan.skip_frames_start},{self.scan.experimental_parameters.number_frames_raw - self.scan.skip_frames_start})\n')
            f.write('all_g2.npy:\n')
            f.write('- Contains g2-functions for the q-partitions specified.\n')
            f.write(f'- Shape (number of q-partitions, number of frames): ({self.results_TTC.q_partitions.number_q_partitions},{self.scan.experimental_parameters.number_frames_raw - self.scan.skip_frames_start-1})\n')
            f.write('- Does not contain the values on the diagonal of the TTCs.\n')
            f.write('- Were obtained from diagonal cuts.\n')
            f.write('## END ##')


class Results_SAXS:
    def __init__(self, precision_SAXS, SAXS_Q, SAXS_IofQ, SAXS_errorIofQ, pyFAIconfig):
        self.precision_SAXS = precision_SAXS
        self.SAXS_Q = SAXS_Q
        self.SAXS_IofQ = SAXS_IofQ
        self.SAXS_errorIofQ = SAXS_errorIofQ
        self.pyFAIconfig = pyFAIconfig

    def plot_SAXS_curve(self):
        plt.figure()
        plt.plot(self.SAXS_Q*1e-9,self.SAXS_IofQ)
        plt.yscale('log')
        plt.xlabel('Q (nm-1)')
        plt.ylabel('Intensity (arb. u.)')
        plt.show()
        
class Results_SAXS_slices: 
    def __init__(self, average_over_frames, precision_SAXS, SAXS_Q, SAXS_IofQ_list, SAXS_errorIofQ_list, pyFAIconfig):
        self.average_over_frames = average_over_frames
        self.precision_SAXS = precision_SAXS
        self.SAXS_Q = SAXS_Q
        self.SAXS_IofQ_list= SAXS_IofQ_list
        self.SAXS_errorIofQ_list = SAXS_errorIofQ_list
        self.pyFAIconfig = pyFAIconfig

    def plot_SAXS_curves(self): 
        plt.figure()
        cmap = [cm.get_cmap('jet')(i / (len(self.SAXS_IofQ_list))) for i in range(len(self.SAXS_IofQ_list))]
        for i in range(len(self.SAXS_IofQ_list)):
            plt.plot(self.SAXS_Q*1e-9,self.SAXS_IofQ_list[i], color=cmap[i])
        plt.yscale('log')
        plt.xlabel('Q (nm-1)')
        plt.ylabel('Intensity (arb. u.)')
        plt.show()

class Results_TTC:
    def __init__(self, intensity_per_q_partition, ttc_per_q_partition, pixels_per_q_partition, q_partitions):
        self.intensity_per_q_partition = intensity_per_q_partition
        self.ttc_per_q_partition = ttc_per_q_partition
        self.pixels_per_q_partition = pixels_per_q_partition
        self.q_partitions = q_partitions
        self.g2_functions = None

    def create_delay_times(self, experimental_parameters):
        self.delay_times = np.arange(0, self.intensity_per_q_partition.shape[1] * experimental_parameters.time_interval_s, experimental_parameters.time_interval_s)

    def plot_intensity_variation(self, experimental_parameters):
        plt.figure()
        for i in range(self.q_partitions.number_q_partitions):
            plt.plot(self.delay_times, self.intensity_per_q_partition[i,:] / np.mean(self.intensity_per_q_partition[i,0:5]) + i, color=self.q_partitions.colors_q_partitions[i])
        plt.ylabel('Average Intensity per q-partition (norm.)')
        plt.xlabel('Time (s)')
        plt.show()

    def range_plot_ttcs(self):
        ttc_data_copy = np.copy(self.ttc_per_q_partition)
        for i in range(self.q_partitions.number_q_partitions):
            np.fill_diagonal(ttc_data_copy[i], np.nan)
        vmin = np.nanpercentile(ttc_data_copy, 40)
        vmax = np.nanpercentile(ttc_data_copy, 60)
        return vmin, vmax

    def plot_ttcs(self):
        plot_vmin, plot_vmax = self.range_plot_ttcs()
        plot_columns = 5
        plot_rows = (self.q_partitions.number_q_partitions // plot_columns) + 1
        subfigure_size = 2
        fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(plot_columns * subfigure_size, plot_rows * subfigure_size))
        axes = axes.flatten()
        for i in range(self.q_partitions.number_q_partitions):
            next_q_center = self.q_partitions.q_rings.q_radii[i]*1e-9
            axes[i].imshow(self.ttc_per_q_partition[i,:,:], vmin=plot_vmin, vmax=plot_vmax, origin='lower')
            axes[i].set_title(f'q = {next_q_center:.3f} nm-1')
        plt.tight_layout()
        plt.show()

    def calculate_g2s_diagonal_cuts(self):
        number_frames_analyzed = len(self.delay_times)
        self.g2_functions = np.zeros((self.q_partitions.number_q_partitions,number_frames_analyzed - 1))
        for i in range(self.q_partitions.number_q_partitions):
            self.g2_functions[i,:] = calculate_g2_diagonal_cut(self.ttc_per_q_partition[i])

    def plot_g2s(self):
        number_frames_analyzed = len(self.delay_times)
        subset_4ylimits = self.g2_functions[:, :-number_frames_analyzed // 5] # take away the final fifth of the data
        plot_min = np.nanmin(subset_4ylimits)
        plot_max = np.nanmax(subset_4ylimits)
        plt.figure()
        for i in range(self.q_partitions.number_q_partitions):
            plt.plot(self.delay_times[1:], self.g2_functions[i])
        plt.xscale('log')
        plt.xlim([self.delay_times[1], self.delay_times[-20]])
        plt.ylim([plot_min,plot_max])
        plt.show()
        
class Results_temp:
    def __init__(self, temp_time_list, temp_set_point_list, temp_sensorA_list, temp_sensorB_list, temp_heater_output_list, temp_range_list): 
        self.temp_time_list = temp_time_list
        self.temp_set_point_list = temp_set_point_list 
        self.temp_sensorA_list = temp_sensorA_list 
        self.temp_sensorB_list = temp_sensorB_list 
        self.temp_heater_output_list = temp_heater_output_list 
        self.temp_range_list =  temp_range_list 

    def plot_temp_data(self):
        plt.figure()
        plt.plot(self.temp_time_list, self.temp_set_point_list, label='Set Point', color='black')
        plt.plot(self.temp_time_list, self.temp_sensorA_list, label='Sensor A', color='red')
        plt.plot(self.temp_time_list, self.temp_sensorB_list, label='Sensor B', color='blue')
        plt.xlabel('time [s]')
        plt.ylabel('temperature [K]')
        plt.legend()
        plt.show()

### ADDITIONAL FUNCTIONS #####
def calculate_ttc(data_2D):
    # calculate variation in intensity per frame
    intensity_change = np.mean(data_2D, axis=1)
    # determine number of pixels
    number_of_pixels = data_2D.shape[1]
    # calculate ttc via dot-product
    tmp_ttc_triangle = scipy.linalg.blas.dsyrk(alpha=1.0, a=data_2D, beta=0.0, trans=False, lower=False)
    tmp_ttc_full = tmp_ttc_triangle + tmp_ttc_triangle.T - np.diag(np.diag(tmp_ttc_triangle))
    ttc = tmp_ttc_full / np.outer(intensity_change, intensity_change) / number_of_pixels
    return intensity_change, number_of_pixels, ttc    

def calculate_g2_diagonal_cut(ttc):
    g2 = np.array([np.nanmean(np.diagonal(ttc, offset=i)) for i in range(1,(ttc.shape[0]))])
    return g2


def  month_three_letters_to_number(month): # is replacable with datetime module (as implemented in scan get temp file data
    "Converts a month given as the first three letters (example: Apr) and returns the number of the month as a string with 2 numbers (from prevoius example: 04"
    months ={"Jan":"01",
             "Feb":"02",
             "Mar":"03",
             "Apr":"04",
             "May":"05",
             "Jun":"06",
             "Jul":"07",
             "Aug":"08",
             "Sep":"09",
             "Oct":"10",
             "Nov":"11",
             "Dec":"12"}
    return months[month]