import os
import struct
import PySimpleGUI as sg
import numpy as np
import xarray as xr
import netCDF4 as ds
from pathlib import Path
import matplotlib.pyplot as plt
import struct
import itertools

class Homoganizor:
    
    def __init__(self):
        self.list_FIDs = []
        self.reconstructed_image_list = []
        self.field_map_list = []
        self.interpolated_field_map = None
        self.gamma = 48.52*10**6
        self.te_array = np.arange(1.6,2.1,0.1)*10**-3 # replace with user_prefs / automatic from dir
        self.delta_te = self.te_array[1] - self.te_array[0] #constant difference # replace with user_prefs / automatic from dir
        self.dimension = [128,128,2] # replace with user_prefs / automatic from dir
        self.submit_button = "Submit"
        self.cancel_button = "Cancel" # remove if not used
        self.scan_folders_path = None
        self.save_path = None
        self.main_window_layout = [
            [sg.Text('Welcome to MRI Reconstruction Tool!', size=(40, 1), font=("Helvetica", 15))],
            [sg.Text("First, browse a folder containing the scan folders of the required data.\n"
                    "Then, choose the calculation you would like to make.\n"
                    "For each calculation, mark 'V' if you would like to save the results and/or display the image.\n")],
            [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
            sg.InputText('Current Folder'), sg.FolderBrowse()],
            [sg.Radio('Calculate reconstructed images', "RADIO1", default=True), sg.Radio('Calculate field maps', "RADIO1"),
            sg.Radio('Calculate interpolated total field map', "RADIO1", default=True)],
            [sg.Checkbox('Save reconstructed images       '), sg.Checkbox('Save field maps       ', default=True), sg.Checkbox('Save interpolated total field map')],
            [sg.Checkbox('Display image                          '), sg.Checkbox('Display image         ', default=True), sg.Checkbox('Display image')],
            [sg.Text('_'  * 80)],
            [sg.Submit(), sg.Cancel()]
            ]
        self.save_popup_layout = [
            [sg.Text("Browse a folder that you would like to save the data in.", size=(50, 1))],
            [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
            sg.InputText('Default Folder'), sg.FolderBrowse()],
            [sg.Submit(), sg.Cancel()]
            ]

    def ReconstractImageFromFID(self, FID):
        pass

    def CreateFieldMapFromImages(self, image1, imagae2):
        pass

    def InterpolateFieldMap(self, field_maps):
        pass

    # def Capsule 1

    # def Capsule 2 CreateFieldMapFromFIDs
    
    # def Capsule 3 

    def display_image(self, img):
        '''
        Displays the reconstructed image
        ''' 
        #abs_img    = abs(img)
        plt.imshow(img)
        plt.show()

    def compute_phase(self, img1,img2):
        '''
        Gets two reconstructed images and computes one phase image
        '''
        conj_img2           = np.conj(img2)
        multiplic_img1_img2 = conj_img2*img1
        phase_map           = np.angle(multiplic_img1_img2)

        return phase_map

    def save_arrays_to_disk(self, path, arrays_dictionary: dict, file_name: str):
        """
        Convert every numpy array in arrays_dictionary to xarray and save it in the given path as a NetCDF file.
        """
        for key, value in arrays_dictionary.items():
            arr = value
            x_arr = xr.DataArray(arr)
            name = file_name + str(key)
            x_arr.to_netcdf(f'{path}\\{name}.nc', mode='w')

    def compute_field_values(self, phase_map):
        '''
        Gets a phase image and computes field map
        '''
        bmap = phase_map/((2*np.pi*self.gamma*(self.delta_te)))
        
        return bmap    

    def reconstruct_image(self, fid_arr,dimension):
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

    def reconstruct_images(self, fid_list):
        reconstructed_image_list = []    
        for fid in fid_list:
            reconstructed_image_list.append(self.reconstruct_image(fid, self.dimension))
            
        return reconstructed_image_list    

    def calc_field_maps_from_fids (self, list_of_fids, dimension):
        ''' Gets list of FID files and calculates list of field maps
            by running on pairs of FID files
        '''
        list_of_field_maps = []    
        reconstructed_image_list = self.reconstruct_images(list_of_fids)
        image_pairs = self.pairwise(reconstructed_image_list)
        for img1, img2 in image_pairs:
            phase_map = self.compute_phase(img1,img2)
            bmap      = self.compute_field_values(phase_map)
            list_of_field_maps.append(bmap)
        return list_of_field_maps

    def pairwise(self, list_of_arrays):
        '''
        Creates pairs of FID files from list of FIDs
        list_of_fids -> (fid0,fid1), (fid1,fid2), (fid2, fid3), and so forth...

        '''
        array1, array2 = itertools.tee(list_of_arrays)
        next(array2, None)
        return zip(array1, array2)
        
    def interpolate_field_map(self, list_of_field_maps,te,dimension,signals_amount):
        '''
        Calculates one interpolated field map from all the calculated field maps
        '''
        slope=np.zeros((dimension[0],dimension[1]))
        value_vec_in_phase_map = np.zeros(len(list_of_field_maps))
        for x in range(dimension[0]-1):
            for y in range(dimension[1]-1):
                for z in range(signals_amount-1):
                    value_vec_in_phase_map[z] = list_of_field_maps[z][x,y]
                s,intercept = np.polyfit((te[:]),value_vec_in_phase_map,1)
                slope[x,y] = (s)
        interp_b=slope/self.gamma
        return interp_b

    def interpolate_field_map_from_fids(self, list_of_fids):
        '''
        Gets list of FID files and calculates one interpolated field map 
        '''
        signals_amount = len(list_of_fids)
        return self.interpolate_field_map(self.calc_field_maps_from_fids(list_of_fids,self.dimension), self.te_array, self.dimension,signals_amount)  

    def save_results(self):
        pass

    def create_fid_list(self, folder_path):
    
        dir_list = os.listdir(folder_path)
        for scan_dir in dir_list:
            file_path = folder_path + '\\' + scan_dir
            if os.path.isdir(file_path):
                fid_path = self.find_fid(file_path)
                if isinstance(fid_path, str):
                    self.list_FIDs.append(self.fid_to_nparray(fid_path))

    def find_fid(self, containing_folder):
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
        # Add other initialization?
        window = sg.Window('MRI Reconstruction Tool', default_element_size=(40, 1)).Layout(self.main_window_layout)
        button, user_prefs = window.Read()
        window.Close()
                
        if button == self.submit_button:

            if user_prefs[0] == 'Current Folder':
                scan_folders_path = os.getcwd()
            else:
                scan_folders_path = user_prefs[0]

            self.create_fid_list(scan_folders_path)
    
            if user_prefs[4] or user_prefs[5] or user_prefs[6]:
                window = sg.Window('Choose folder to save files to', default_element_size=(40, 1)).Layout(self.save_popup_layout)
                button, save_prefs = window.Read()
                if button != self.submit_button:
                    raise SystemExit
                if save_prefs[0] == 'Current Folder':
                    self.save_folder = scan_folders_path
                else:
                    self.save_folder = save_prefs[0] + '\\Results' #make sure to create if does not exist

            if user_prefs[1]:
                self.reconstructed_image_list = self.reconstruct_images(self.list_FIDs)
                if user_prefs[4]:
                    pass    # call self.save_func()
                if user_prefs[7]:
                    pass    # show RI images() ABSSSSSSS
            else:
                if user_prefs[2]:
                    self.field_map_list = self.calc_field_maps_from_fids(self.list_FIDs, self.dimension)
                            # call self.save_func()
                    if user_prefs[7]:
                        pass    # show RI images()
                    if user_prefs[8]:
                        pass    # show field map images()
                else:
                    self.interpolated_field_map = self.interpolate_field_map_from_fids(self.list_FIDs)
                            # call self.save_func()
                    if user_prefs[7]:
                        pass    # show RI images()
                    if user_prefs[8]:
                        pass    # show field map images()
                    if user_prefs[9]:
                        pass    # show interpolate field map image()



if __name__ == "__main__":
    homoganizor = Homoganizor()
    homoganizor.start()
    #homoganizor.CreateListFIDs("C:\\Users\\User_PC\\Documents\\Hackathon\\Database\\Homogeneous")
    #print(homoganizor.list_FIDs)
