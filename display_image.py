def display_image(list_of_imgs):
    '''
    Displays the reconstructed image
    ''' 
    #abs_img    = abs(img)
    for image in list_of_imgs:
        plt.imshow(image)
        plt.show()
