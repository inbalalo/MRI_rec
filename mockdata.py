import pickle

fid_dict = pickle.load(open("fid_dict.dat","rb"))
dimension = pickle.load(open("dimensions.dat","rb")) 
fid_arr = pickle.load(open("fid_arr.dat","rb"))
img1 = pickle.load(open("img1.dat","rb"))
img2 = pickle.load(open("img2.dat","rb"))
object_list = pickle.load(open("object_list.dat","rb"))
field_maps_list = pickle.load(open("field_maps_list.dat","rb"))
te_values = pickle.load(open("te_values.dat","rb")) 
signals_amount = pickle.load(open("signals_amoung.dat","rb"))
folder_path = pickle.load(open("folder_path.dat","rb"))
fid_path = pickle.load(open("fid_path.dat","rb"))
name_string = pickle.load(open("name_string.dat","rb"))
containing_folder = pickle.load(open("containing_folder.dat","rb"))

