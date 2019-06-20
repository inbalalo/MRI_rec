def get_dimension(self, folder_path):
    
    dir_list = os.listdir(self.folder_path)
    for scan_dir in dir_list:
        file_path = self.folder_path + '\\' + scan_dir
        if os.path.isdir(file_path):
            method_path = self.find_fid(file_path)

    with open(method_path, mode='rb') as file: 
        method_r = file.read()
        f=method_r.find(b'PVM_Matrix=( 2 )\n')
        dimension_locked=method_r[f+17:f+24]
    arr=np.zeros(3)
    arr[0]=(str(dimension_locked)[2:5])
    arr[0]=int(arr[0])
    arr[1]=(str(dimension_locked)[6:9])
    arr[1]=int(arr[1])
    arr[2]= 2
    return arr