def compute_phase(self, img1,img2):
    '''
    Gets two reconstructed images and computes one phase image
    '''
    conj_img2           = np.conj(img2)
    if  (img1.shape[1] == img2.shape[0]):
        multiplic_img1_img2 = conj_img2*img1
        phase_map           = np.angle(multiplic_img1_img2)
        return phase_map
    else:
        raise IndexError('Size of matrices not suitable for linear multiplication')