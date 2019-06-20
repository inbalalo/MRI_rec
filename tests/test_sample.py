import os
import Homogenizer
import Homogenizer_GUI

class Test:
 
    def test_reconstruct_image_fid_arr_input(self):
        Homogeniz_instance.scan_folders_path = os.getcwd +'\\database'
        Homogeniz_instance.get_dimensions(Homogeniz_instance.scan_folders_path)
        Homogeniz_instance.create_fid_dict(Homogeniz_instance.scan_folders_path)
        Homogeniz_instance.reconstruct_image(list(Homogeniz_instance.fid_dicrt.values()), Homogeniz_instance.dimensions)
        assert isinstance(list(Homogeniz_instance.fid_dicrt.values(), np.ndarray)

    # We ran out of time... !
    def test_reconstruct_image_dimension_input(self):
        Homogeniz_instance.reconstruct_image(fid_arr, dimension)
        assert isinstance(fid_arr,np.ndarray)

    def test_reconstruct_image_dimension_output(self):
        img = Homogeniz_instance.reconstruct_image(fid_arr, dimension)
        assert isinstance(img,np.ndarray) 

    def test_reconstruct_image_complex_val_suitable_to_reshape(self):
        img = Homogeniz_instance.reconstruct_image(fid_arr, dimension)
        assert len(fid_arr) == dimension[0]*dimension[1]*2             

    def test_calc_field_maps_from_fids_fid_dict_input (self):
        Homogeniz_instance.calc_field_maps_from_fids (fid_dict, dimension)
        assert isinstance(fid_dict,dict)

    def test_calc_field_maps_from_fids_dimension_input (self):
        Homogeniz_instance.calc_field_maps_from_fids (fid_dict, dimension)
        assert isinstance(dimension,np.ndarray) 

    def test_calc_field_maps_from_fids_field_map_dict_input (self):
        Homogeniz_instance.calc_field_maps_from_fids (fid_dict, dimension)
        assert isinstance(field_map_dict,dict) 

    def test_calc_field_maps_from_fids_field_map_dict_vals_are_dict(self):
        for value in field_map_dict.value():
            assert isinstance(value,np.ndarray)

    def test_calc_field_map_from_reconstructed_images_img1_input(self):
        Homogeniz_instance.calc_field_map_from_reconstructed_images(img1,img2)
        assert isinstance(img1,np.ndarray)

    def test_calc_field_map_from_reconstructed_images_img2_input(self):
        Homogeniz_instance.calc_field_map_from_reconstructed_images(img1,img2)
        assert isinstance(img2,np.ndarray)

    def test_calc_field_map_from_reconstructed_images_delta_te_type(self):
        Homogeniz_instance.calc_field_map_from_reconstructed_images(img1,img2)
        assert (isinstance(Homogeniz_instance.delta_te,float) or isinstance(Homogeniz_instance.delta_te,int))

    def test_calc_field_map_from_reconstructed_images_output(self):
        bmap = calc_field_map_from_reconstructed_images(img1,img2)
        assert isinstance(bmap,np.ndarray)

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
  
    def test_interpolate_field_map_from_fids_input(self):
        Homogeniz_instance.interpolate_field_map_from_fids(fid_dict)
        assert isinstance(fid_dict,dict)

    def test_interpolate_field_map_input_field_maps(self): 
        Homogeniz_instance.interpolate_field_map_input_field_maps_list(field_maps_list, te_values, dimension, signals_amount)
        assert isinstance(field_maps_list,list)

    def test_interpolate_te_values_input_te_values(self): 
        Homogeniz_instance.interpolate_field_map_input_field_maps_list(field_maps_list, te_values, dimension, signals_amount)
        assert isinstance(te_values,np.ndarray)

    def test_interpolate_field_map_input_dimension(self): 
        Homogeniz_instance.interpolate_field_map_input_field_maps_list(field_maps_list, te_values, dimension, signals_amount)
        assert isinstance(dimension,np.ndarray)

    def test_interpolate_field_map_input_signals_amount(self): 
        Homogeniz_instance.interpolate_field_map_input_field_maps_list(field_maps_list, te_values, dimension, signals_amount)
        assert isinstance(signlas_amount,int)

    def test_interpolate_field_map_input_interpolated_field_map(self): 
        Homogeniz_instance.interpolate_field_map_input_field_maps_list(field_maps_list, te_values, dimension, signals_amount)
        assert isinstance(interpolated_field_map.interpolated_field_map,np.ndarray)

    def test_create_fid_dict_folder_path_input(self):
        Homogeniz_instance.create_fid_dict(folder_path)
        assert isinstance(folder_path,str)

    def test_find_fid_containing_folder_input(self):
        Homogeniz_instance.find_fid(containing_folder)
        assert isinstance(containing_folder,str)       

    def test_find_fid_containing_folder_output(self):
        file_path=Homogeniz_instance.find_fid(containing_folder)
        assert isinstance(file_path,str)  

    def test_fid_to_nparray_input(self):
        Homogeniz_instance.fid_to_nparray(fid_path)
        assert isinstance(fid_path,str)

    def test_fid_to_nparray_output(self):
        fid_arr = Homogeniz_instance.fid_to_nparray(fid_path)
        assert isinstance(fid_arr,np.ndarray)

if __name__ == '__main__':
    Homogeniz_instance = Homogenizer()

    ttests = Test()
    methods = ["test_reconstruct_image_fid_arr_input","test_reconstruct_image_dimension_input","test_reconstruct_image_dimension_output","test_reconstruct_image_complex_val_suitable_to_reshape","test_calc_field_maps_from_fids_fid_dict_input","test_calc_field_maps_from_fids_dimension_input","test_calc_field_maps_from_fids_field_map_dict_input","test_calc_field_maps_from_fids_field_map_dict_vals_are_dict","test_calc_field_map_from_reconstructed_images_img1_input","test_calc_field_map_from_reconstructed_images_img2_input","test_calc_field_map_from_reconstructed_images_delta_te_type",
    "test_calc_field_map_from_reconstructed_images_output","test_compute_phase_img1_input",
    "test_compute_phase_img2_input","test_compute_phase_validate_matrix_size_suitable_for_multiplication","test_pairwise_input",
    "test_interpolate_field_map_from_fids_input","test_interpolate_field_map_input_field_maps","test_interpolate_te_values_input_te_values","test_interpolate_field_map_input_dimension","test_interpolate_field_map_input_signals_amount","test_interpolate_field_map_input_interpolated_field_map","test_create_fid_dict_folder_path_input","test_find_fid_containing_folder_input","test_find_fid_containing_folder_output","test_fid_to_nparray_input,test_fid_to_nparray_output"]
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
