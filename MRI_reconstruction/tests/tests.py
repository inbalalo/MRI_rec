from MRI_rec.MRI_reconstruction.Homogenizer import *
import os
import numpy as np
from mockdata import *

class Test:


    def test_reconstruct_image_dimension_input(self):
        Homogeniz_instance.reconstruct_image(fid_arr, dimension)
        assert isinstance(dimension,np.ndarray)

    def test_reconstruct_image_dimension_output(self):
        img = Homogeniz_instance.reconstruct_image(fid_arr, dimension)
        assert isinstance(img,np.ndarray) 

    def test_reconstruct_image_complex_val_suitable_to_reshape(self):
        img = Homogeniz_instance.reconstruct_image(fid_arr, dimension)
        assert len(fid_arr) == dimension[0]*dimension[1]*2             

    def test_reconstruct_image_fid_arr_input(self):
        Homogeniz_instance.reconstruct_image(fid_arr, dimension)
        assert isinstance(fid_arr,np.ndarray)

    def test_compute_phase_img1_input(self):
        Homogeniz_instance.compute_phase(img1,img2)
        assert isinstance(img1,np.ndarray)

    def test_compute_phase_img2_input(self):
        Homogeniz_instance.compute_phase(img1,img2)
        assert isinstance(img2,np.ndarray)

    def test_compute_phase_validate_matrix_size_suitable_for_multiplication(self):
        Homogeniz_instance.compute_phase(img1,img2)
        assert img1.shape[1] == img2.shape[0]

    def test_pairwise_input(self):
        Homogeniz_instance.pairwise(object_list)
        assert isinstance(object_list,list)

    def test_interpolate_field_map_input_field_maps(self): 
        Homogeniz_instance.interpolate_field_map(field_maps_list, te_values, dimension, signals_amount)
        assert isinstance(field_maps_list,list)

    def test_create_fid_dict_folder_path_input(self):
        Homogeniz_instance.create_fid_dict(folder_path)
        assert isinstance(folder_path,str)

    def test_find_fid_containing_folder_input(self):
        Homogeniz_instance.find_file_by_name(containing_folder,name_string)
        assert isinstance(containing_folder,str)       

    def test_find_fid_containing_folder_output(self):
        file_path=Homogeniz_instance.find_file_by_name(containing_folder,name_string)
        assert isinstance(file_path,str)  

    def test_fid_to_nparray_input(self):
        Homogeniz_instance.fid_to_nparray(fid_path)
        assert isinstance(fid_path,str)

    def test_fid_to_nparray_output(self):
        fid_arr = Homogeniz_instance.fid_to_nparray(fid_path)
        assert isinstance(fid_arr,np.ndarray)

    def test_calc_field_maps_from_fids_fid_dict_input (self):
        Homogeniz_instance.calc_field_maps_from_fids (fid_dict,dimension)
        assert isinstance(fid_dict,dict)

    def test_calc_field_maps_from_fids_dimension_input (self):
        Homogeniz_instance.calc_field_maps_from_fids (fid_dict,dimension)
        assert isinstance(dimension,np.ndarray) 


    def test_calc_field_map_from_reconstructed_images_delta_te_type(self):
        Homogeniz_instance.calc_field_map_from_reconstructed_images(img1,img2)
        assert (isinstance(Homogeniz_instance.delta_te,float) or isinstance(Homogeniz_instance.delta_te,int))

  
    def test_calc_field_map_from_reconstructed_images_img1_input(self):
        Homogeniz_instance.calc_field_map_from_reconstructed_images(img1,img2)
        assert isinstance(img1,np.ndarray)

    def test_calc_field_map_from_reconstructed_images_img2_input(self):
        Homogeniz_instance.calc_field_map_from_reconstructed_images(img1,img2)
        assert isinstance(img2,np.ndarray)

    def test_calc_field_map_from_reconstructed_images_output(self):
        bmap =  Homogeniz_instance.calc_field_map_from_reconstructed_images(img1,img2)
        assert isinstance(bmap,np.ndarray)

    def test_interpolate_field_map_from_fids_input(self):
        Homogeniz_instance.interpolate_field_map_from_fids(fid_dict)
        assert isinstance(fid_dict,dict)


if __name__ == '__main__':
    Homogeniz_instance = Homogenizer()

    ttests = Test()
    methods = ["reconstruct_image_dimension_input","reconstruct_image_dimension_output","reconstruct_image_complex_val_suitable_to_reshape","reconstruct_image_fid_arr_input","compute_phase_img1_input", "compute_phase_img2_input","compute_phase_validate_matrix_size_suitable_for_multiplication","pairwise_input","create_fid_dict_folder_path_input","find_fid_containing_folder_input","find_fid_containing_folder_output","fid_to_nparray_input","fid_to_nparray_output","calc_field_maps_from_fids_fid_dict_input","calc_field_maps_from_fids_dimension_input","calc_field_map_from_reconstructed_images_delta_te_type","calc_field_map_from_reconstructed_images_img1_input","calc_field_map_from_reconstructed_images_img2_input",
    "calc_field_map_from_reconstructed_images_output"]
    errors = []

    for method in methods:
        try:
            getattr(ttests, "test_" + method)()
        except AssertionError as e:
            errors.append(f"Failed when testing method 'test_{method}': {e}")
            
    if len(errors) > 0:
        print(errors)
    else:
        print("Tests pass successfully.")
