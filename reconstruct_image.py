    def reconstruct_image(self, fid_arr, dimension):
        '''
        Calculates the K space matrix -> calculates the 
        reconstructed image and returns it
        ''' 
        real_vals     = fid_arr[:-1:2]
        imag_vals     = fid_arr[1::2]
        complex_vals  = real_vals + 1j*imag_vals
        if (len(fid_arr) == dimension[0]*dimension[1]*2):
            k_space_scan  = np.reshape(complex_vals,(dimension[0],dimension[1]))
            k_casting     = k_space_scan.astype(complex)
            img           = np.fft.fftshift(np.fft.ifft2(k_casting)) 
            return img
        else:
            raise IndexError('Fid_arr cannot be reshaped to these dimensions')