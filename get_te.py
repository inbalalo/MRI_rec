def get_te(self, folder_path):
    
    dir_list = os.listdir(self.folder_path)
    for scan_dir in dir_list:
        file_path = self.folder_path + '\\' + scan_dir
        if os.path.isdir(file_path):
            method_path = self.find_fid(file_path)

    with open(method_path, mode='rb') as file: # b is important -> binary
        method_r = file.read()
        f=method_r.find(b'EchoTime=')
        te_locked=method_r[f+9:f+12]
        te_str=str(te_locked)[2:5]

    if (str(te_str).find('n') != -1): 
        te=int(te_str[0])
    else:
        te=float(te_str)
    return te