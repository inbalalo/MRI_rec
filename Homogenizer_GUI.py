import PySimpleGUI as sg
import os
    
class Homogenizer_GUI:
    def __init__(self):
        self.current_folder_expression = 'Current Folder'
        self.default_folder_expression = 'Default Folder'
        self.user_prefs = None
        self.last_button_pressed = None
        self.main_window_layout = None
        self.save_popup_layout = [
            [sg.Text("Browse a folder that you would like to save the data in.", size=(50, 1))],
            [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
            sg.InputText(self.default_folder_expression), sg.FolderBrowse()],
            [sg.Submit(), sg.Cancel()]
            ]

    def set_main_layout(self):
        self.main_window_layout = [
            [sg.Text('Welcome to MRI Reconstruction Tool!', size=(40, 1), font=("Helvetica", 15))],
            [sg.Text("First, browse a folder containing the scan folders of the required data.\n"
                    "Then, choose the calculation you would like to make.\n"
                    "For each calculation, mark 'V' if you would like to save the results and/or display the image.\n")],
            [sg.Text('Your Folder', size=(15, 1), auto_size_text=False, justification='right'),
            sg.InputText(self.current_folder_expression), sg.FolderBrowse()],
            [sg.Radio('Calculate reconstructed images', "RADIO1"), sg.Radio('Calculate field maps', "RADIO1"),
            sg.Radio('Calculate interpolated total field map', "RADIO1", default=True)],
            [sg.Checkbox('Save reconstructed images       ', default=True), sg.Checkbox('Save field maps       ', default=True), sg.Checkbox('Save interpolated total field map', default=True)],
            [sg.Checkbox('Display image                          '), sg.Checkbox('Display image         '), sg.Checkbox('Display image')],
            [sg.Text('_'  * 80)],
            [sg.Submit(), sg.Cancel()]
            ]

    def open_main_window(self):
        self.set_main_layout()
        window = sg.Window('MRI Reconstruction Tool', default_element_size=(40, 1)).Layout(self.main_window_layout)
        self.last_button_pressed, self.user_prefs = window.Read()
        window.Close()
        if self.user_prefs[0] == self.current_folder_expression:
            return os.getcwd()
        return self.user_prefs[0]

    def request_save_path(self):
        window = sg.Window('Choose folder to save files to', default_element_size=(40, 1)).Layout(self.save_popup_layout)
        self.last_button_pressed, save_prefs = window.Read()
        return save_prefs[0]
