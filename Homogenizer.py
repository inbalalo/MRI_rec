import os
import struct
import numpy as np
import xarray as xr
import netCDF4 as ds
from pathlib import Path
import matplotlib.pyplot as plt
import struct
import itertools
import Homogenizer_GUI
from enum import Enum
from collections import OrderedDict

class UserPrefs(Enum):
    ScanFoldersPath = 0
    CalculateReconstructedImages = 1
    CalculateFieldMaps = 2
    CalculateInterpolatedFieldMap = 3
    SaveReconstructedImages = 4
    SaveFieldMaps = 5
    SaveInterpolatedFieldMap = 6
    ShowReconstructedImages = 7
    ShowFieldMaps = 8
    ShowInterpolatedFieldMap = 9
    DimensionX = 10
    DimensionY = 11
    DimensionZ = 12

class Homogenizer: 
    def __init__(self):
        self.hGUI = None
        self.submit_button = "Submit"
        self.gamma = 48.52*10**6
        self.te_array = np.arange(1.6,2.1,0.1)*10**-3 # replace with user_prefs / automatic from dir
        self.delta_te = self.te_array[1] - self.te_array[0] #constant difference # replace with user_prefs / automatic from dir
        self.dimension = [128,128,2] # replace with user_prefs / automatic from dir
        self.scan_folders_path = None
        self.save_path = None
        self.fids_dict = OrderedDict([])
        self.reconstructed_image_dict = OrderedDict([])
        self.field_map_dict = OrderedDict([])
        self.interpolated_field_map = OrderedDict([])

    def get_input(self, user_pref: UserPrefs):
        return self.hGUI.user_prefs[user_pref.value]

    def display_image(self, img):
        '''
        Displays the reconstructed image
        ''' 
        plt.imshow(img)
        plt.show()

    def save_arrays_to_disk(self, path, arrays_dictionary: dict, name_prefix: str):
        """
        Convert every numpy array in arrays_dictionary to xarray and save it in the given path as a NetCDF file.
        """
        for key, array in arrays_dictionary.items():  #make sure to create folder if does not exist
            x_arr = xr.DataArray(array)
            file_name = name_prefix + str(key)
            x_arr.to_netcdf(f'{path}\\{file_name}.nc', mode='w')

    def reconstruct_images_from_fids(self, fid_dict):
        for name_prefix, fid in fid_dict.items():
            self.reconstructed_image_dict[name_prefix] = self.reconstruct_image(fid, self.dimension)

    def reconstruct_image(self, fid_arr, dimension):
        '''
        Calculates the K space matrix -> calculates the 
        reconstructed image and returns it
        ''' 
        real_vals     = fid_arr[:-1:2]
        imag_vals     = fid_arr[1::2]
        complex_vals  = real_vals + 1j*imag_vals
        k_space_scan  = np.reshape(complex_vals,(dimension[0],dimension[1]))
        k_casting     = k_space_scan.astype(complex)
        img           = np.fft.fftshift(np.fft.ifft2(k_casting)) 
        return img

    def calc_field_maps_from_fids (self, fid_dict, dimension):
        ''' Gets an ordered dictionary of FID files and calculates dictionary of field maps
            by running on pairs of FID files
        '''
        self.reconstruct_images_from_fids(fid_dict)
        image_pairs = self.pairwise(self.reconstructed_image_dict.values())
        name_index = 0
        name_list = list(self.reconstructed_image_dict.keys())
        for img1, img2 in image_pairs:
            field_map_prefix = name_list[name_index] + name_list[name_index+1]
            name_index +=1
            self.field_map_dict[field_map_prefix] = self.calc_field_map_from_reconstructed_images(img1,img2)

    def calc_field_map_from_reconstructed_images(self, img1,img2):
            phase_map = self.compute_phase(img1,img2)
            bmap = phase_map/((2*np.pi*self.gamma*(self.delta_te)))
            return bmap

    def compute_phase(self, img1,img2):
        '''
        Gets two reconstructed images and computes one phase image
        '''
        conj_img2           = np.conj(img2)
        multiplic_img1_img2 = conj_img2*img1
        phase_map           = np.angle(multiplic_img1_img2)
        return phase_map

    def pairwise(self, object_list):
        '''
        Creates pairs of objects from a list of objects
        list_of_fids -> (fid0,fid1), (fid1,fid2), (fid2, fid3), and so forth...

        '''
        obj1, obj2 = itertools.tee(object_list)
        next(obj2, None)
        return zip(obj1, obj2)

    def interpolate_field_map_from_fids(self, fid_dict):
        '''
        Gets an ordered dictionary of FID files and calculates one interpolated field map 
        '''
        signals_amount = len(fid_dict)
        self.calc_field_maps_from_fids(fid_dict, self.dimension)
        self.interpolate_field_map(list(self.field_map_dict.values()), self.te_array, self.dimension,signals_amount) 
        
    def interpolate_field_map(self, field_maps_list, te_values, dimension, signals_amount):
        '''
        Calculates one interpolated field map from all the calculated field maps
        '''
        slope=np.zeros((dimension[0],dimension[1]))
        value_vec_in_phase_map = np.zeros(len(field_maps_list))
        for x in range(dimension[0]-1):
            for y in range(dimension[1]-1):
                for z in range(signals_amount-1):
                    value_vec_in_phase_map[z] = field_maps_list[z][x,y]
                s,intercept = np.polyfit((te_values[:]),value_vec_in_phase_map,1)
                slope[x,y] = (s)
        interp_b=slope/self.gamma
        self.interpolated_field_map = OrderedDict([('',interp_b)])

    def create_fid_dict(self, folder_path):
        '''
        Creates an ordered dictionary of numpy arrays from fid files
        '''
        dir_list = os.listdir(folder_path)
        for scan_dir in dir_list:
            file_path = folder_path + '\\' + scan_dir
            if os.path.isdir(file_path):
                fid_path = self.find_fid(file_path)
                if isinstance(fid_path, str):
                    self.fids_dict[scan_dir] = self.fid_to_nparray(fid_path)

    def find_fid(self, containing_folder):
        '''
        Finds and returns the fid file within the given folder
        '''
        dir_list = os.listdir(containing_folder)
        for file_name in dir_list:
            if file_name == "fid":
                file_path = containing_folder + '\\' + file_name
                return file_path

    def fid_to_nparray(self, fid_path):
        '''
        Opens a binary file and inserts it to a numpy array
        ''' 
        with open(fid_path, mode='rb') as file: # b is important -> binary
            fid_r = file.read()
            fid_l = list(struct.unpack("i" * ((len(fid_r) -4) // 4), fid_r[0:-4]))
            fid_l.append(struct.unpack("i", fid_r[-4:])[0])
            fid_arr = np.array(fid_l)
        return fid_arr

    def start(self):
        self.scan_folders_path = self.hGUI.open_main_window()
        # Starts job if user had pressed submit:     
        if self.hGUI.last_button_pressed == self.submit_button:
            # Checks if user requested to save any files, and if so pops up a browser to choose path.
            if (self.get_input(UserPrefs.SaveReconstructedImages)
            or self.get_input(UserPrefs.SaveFieldMaps) 
            or self.get_input(UserPrefs.SaveInterpolatedFieldMap)
            ):
                self.save_path = self.hGUI.request_save_path()
                # Cancels the job if the user had closed the window / pressed "Cancel":
                if self.hGUI.last_button_pressed != self.submit_button:
                    self.start()
                    return
                if self.save_path == self.hGUI.default_folder_expression:
                    self.save_path = self.scan_folders_path + '\\Results'
            self.create_fid_dict(self.scan_folders_path)
            # Checks what calculation the user had requested, and performs them:
            if self.get_input(UserPrefs.CalculateReconstructedImages):
                self.reconstruct_images_from_fids(self.fids_dict)
            else:
                if self.get_input(UserPrefs.CalculateFieldMaps):
                    self.calc_field_maps_from_fids(self.fids_dict, self.dimension)
                else:
                    self.interpolate_field_map_from_fids(self.fids_dict)
                    if self.get_input(UserPrefs.SaveInterpolatedFieldMap):
                        self.save_arrays_to_disk(self.save_path, self.interpolated_field_map,'Interpolated_field_map')
                    if self.get_input(UserPrefs.ShowInterpolatedFieldMap):
                        pass    # show interpolate field map image()
                if self.get_input(UserPrefs.SaveFieldMaps):
                    self.save_arrays_to_disk(self.save_path, self.field_map_dict, 'Field_map_')
                if self.get_input(UserPrefs.ShowFieldMaps):
                    pass    # show field map images()
            if self.get_input(UserPrefs.SaveReconstructedImages):
                self.save_arrays_to_disk(self.save_path, self.reconstructed_image_dict, 'Reconstructed_Image_')
            if self.get_input(UserPrefs.ShowReconstructedImages):
                pass    # show RI images() ABSSSSSSS

if __name__ == "__main__":
    homogenizer = Homogenizer()
    homogenizer.hGUI = Homogenizer_GUI.Homogenizer_GUI()
    homogenizer.start()
