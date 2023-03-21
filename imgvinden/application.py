import sys
import os
import math

from .dialogues import *
from .utils import get_random_string

import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import numpy as np
from PIL import Image

# Image class used to show contents
class ImageVinden(QPixmap):
    # TODO: make it so the images are not squashed when in frame. Add zero padding...
    default_image_path = "./imgvinden/images/default_image.jpg"
    def __init__(self, path = None):
        if path == None:
            self.image_path = self.default_image_path
            super().__init__(self.default_image_path)
        else:
            self.image_path = path
            super().__init__(path)

        self.original_image_path = self.image_path

        # Making the directory we will put all of the generated images
        self.dir_name = self.image_path.split("/")[-1].split(".")[0]
        os.makedirs(os.path.join("./imgvinden/images/", self.dir_name), exist_ok=True)

        self.image_save_counter = 0

    def load_new_image(self, fileName : str):
        self.original_image_path = fileName
        self.load(fileName=fileName)

    def load(self, fileName : str):
        super().load(fileName)

        self.image_path = fileName
        self.dir_name = self.image_path.split("/")[3].split(".")[0].split("-K:")[0]
        os.makedirs(os.path.join("./imgvinden/images/", self.dir_name), exist_ok=True)

    def image_to_np(self):
        with Image.open(self.original_image_path) as im:
            return np.asarray(im)
        
    def use_np_image(self, image : np.ndarray, save_name : str):
        # Convert to PIL
        im = Image.fromarray(image)
        image_path = os.path.join("./imgvinden/images/", self.dir_name, save_name + "_" + str(self.image_save_counter).zfill(3) + ".jpg")
        im.save(image_path)

        self.image_save_counter += 1

        self.load(image_path)

    def flip(self, flip_direction : str):
        assert flip_direction in ["horizontal", "vertical"]

        image = self.image_to_np()
        flipped_image = np.copy(image)

        height, width, _ = image.shape

        for i in range(height):
            for j in range(width):
                if flip_direction == "horizontal":
                    flipped_image[i, j, :] = image[i, width - j - 1, :]
                else:
                    flipped_image[i, j, :] = image[height - i - 1, j, :]

        self.use_np_image(flipped_image, save_name="flipped")

    def rotate(self, rotation_degrees : int):
        rotation_rad = math.radians(rotation_degrees)
        
        def rotate_point(x, y, alpha):
            x_prime = x*math.cos(alpha) - y*math.sin(alpha)
            y_prime = x*math.sin(alpha) + y*math.cos(alpha)
            return (x_prime, y_prime)

        image = self.image_to_np()

        height, width, _ = image.shape
        
        # Getting the square area of the rotated image when zero padding is used
        tl = rotate_point(0, 0, alpha = rotation_rad)
        bl = rotate_point(height - 1, 0, alpha = rotation_rad)
        tr = rotate_point(0, width - 1, alpha = rotation_rad)
        br = rotate_point(height - 1, width - 1, alpha = rotation_rad)

        x_min = min(tl[0], bl[0], tr[0], br[0])
        x_max = max(tl[0], bl[0], tr[0], br[0])

        y_min = min(tl[1], bl[1], tr[1], br[1])
        y_max = max(tl[1], bl[1], tr[1], br[1])

        x_range = int(x_max - x_min) + 1
        y_range = int(y_max - y_min) + 1

        rotated = np.zeros(shape = [x_range + 4, y_range + 4, 3], dtype=np.uint8)

        # TODO add bilinear interpolation?? Might be fun

        for x in range(height):
            for y in range(width):
                x_prime, y_prime = rotate_point(x, y, alpha = rotation_rad)

                x_prime = int(x_prime + 0.5) - int(x_min) + 2
                y_prime = int(y_prime + 0.5) - int(y_min) + 2

                rotated[x_prime, y_prime, :] = image[x, y, :]

        self.use_np_image(rotated, save_name="rotate")

    def scale(self, scale_factor : float):
        image = self.image_to_np()

        height, width, _ = image.shape

        out_image = np.zeros(shape = [int(height * scale_factor) + 50, int(width * scale_factor) + 50, 3], dtype = np.uint8)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                x = float(x)
                y = float(y)
                # TODO: Implement bilear interpolation
                x_prime = int(max(0, min(out_image.shape[0] - 1, x * scale_factor)))
                y_prime = int(max(0, min(out_image.shape[1] - 1, y * scale_factor)))

                print(x_prime, y_prime)

                out_image[x_prime, y_prime, :] = image[int(x), int(y), :]

        self.use_np_image(out_image, save_name = "scale")

class ImgVindenGUI(QMainWindow):
    def __init__(self):
        super().__init__()
 
        # set the title
        self.setWindowTitle("ImgVinden")
         
        # setting  the fixed width of window
        height = 1000
        width = 2000
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        # Defining the layout
        self.master_grid = QGridLayout()

        # image grid
        self.image_grid = QGridLayout()

        self.input_label = QLabel()
        self.input_pixmap = ImageVinden()
        self.input_label.setPixmap(self.input_pixmap)
        self.input_label.setScaledContents(False)
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setStyleSheet("border: 3px solid black;")


        self.output_label = QLabel()
        self.output_pixmap = ImageVinden()
        self.output_label.setPixmap(self.output_pixmap)
        self.output_label.setScaledContents(False)
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setStyleSheet("border: 3px solid black;")

        self.image_grid.addWidget(self.input_label, 0, 0)
        self.image_grid.addWidget(self.output_label, 0, 1)

        # button grid
        self.button_grid = QGridLayout()

        self.input_image_button = QPushButton("Select Input Image", self)
        self.input_image_button.clicked.connect(self.input_image_action)
        self.button_grid.addWidget(self.input_image_button, 0, 0)

        self.crop_button = QPushButton("Crop", self)
        self.crop_button.clicked.connect(self.crop_action)
        self.button_grid.addWidget(self.crop_button, 1, 0)

        self.flip_button = QPushButton("Flip", self)
        self.flip_button.clicked.connect(self.flip_action)
        self.button_grid.addWidget(self.flip_button, 2, 0)

        self.scale_button = QPushButton("Scale", self)
        self.scale_button.clicked.connect(self.scale_action)
        self.button_grid.addWidget(self.scale_button, 0, 1)

        self.LGLM_button = QPushButton("Linear Grey Level Mapping", self)
        self.LGLM_button.clicked.connect(self.LGLM_action)
        self.button_grid.addWidget(self.LGLM_button, 1, 1)

        self.PLGLM_button = QPushButton("Power-Law Grey Level Mapping", self)
        self.PLGLM_button.clicked.connect(self.PLGLM_action)
        self.button_grid.addWidget(self.PLGLM_button, 2, 1)

        self.convolution_button = QPushButton("Convolution", self)
        self.convolution_button.clicked.connect(self.convolution_action)
        self.button_grid.addWidget(self.convolution_button, 0, 2)

        self.rotate_button = QPushButton("Rotate", self)
        self.rotate_button.clicked.connect(self.rotate_action)
        self.button_grid.addWidget(self.rotate_button, 1, 2)

        self.master_grid.addLayout(self.image_grid, 0, 0)
        self.master_grid.addLayout(self.button_grid, 1, 0)
        self.master_grid.setRowStretch(0, 2)
        self.master_grid.setRowStretch(0, 1)

        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(self.master_grid)

        # show all the widgets
        self.show()

    ##################
    # Button Actions #
    ##################

    @pyqtSlot()
    def input_image_action(self):
        print("input_image_action")

        file_name, ok = QInputDialog.getText(self, 'Image name', 'Input Filename of Image Stored Under imgvinden/images')
        file_path = f"./imgvinden/images/{file_name}"

        if ok:
            if os.path.exists(file_path):
                self.input_pixmap.load_new_image(file_path)
                self.input_label.setPixmap(self.input_pixmap)

                self.output_pixmap.load_new_image(file_path)
                self.output_label.setPixmap(self.output_pixmap)
            else:
                msg = QMessageBox()
                msg.setText(f"Error no file found named \"{file_name}\"")
                msg.exec_()

    @pyqtSlot()
    def crop_action(self):
        print("crop_action")

    @pyqtSlot()
    def flip_action(self):
        print("flip_action")

        msgbox = QMessageBox()
        msgbox.setWindowTitle("Flip Image")
        msgbox.setText('Would you like to flip vertically or horizontally?')
        msgbox.addButton('Vertical', PyQt5.QtWidgets.QMessageBox.NoRole)
        msgbox.addButton('Horizontal', PyQt5.QtWidgets.QMessageBox.NoRole)
        
        if msgbox.exec_():
            flip_direction = "horizontal"
        else:
            flip_direction = "vertical"

        self.output_pixmap.flip(flip_direction = flip_direction)
        self.output_label.setPixmap(self.output_pixmap)

    @pyqtSlot()
    def scale_action(self):
        print("scale_action")

        scale_factor, ok = QInputDialog.getText(self, 'Scale Factor', 'By what scale factor would you like to scale the image? (ex. 2.0)')
        if ok:
            try:
                scale_factor = float(scale_factor)
                self.output_pixmap.scale(scale_factor)
                self.output_label.setPixmap(self.output_pixmap)
            except ValueError:
                msg = QMessageBox()
                msg.setText(f"Error not a valid rotation amount")
                msg.exec_()

    
    @pyqtSlot()
    def LGLM_action(self):
        print("LGLM_action")

    @pyqtSlot()
    def PLGLM_action(self):
        print("PLGLM_action")
    
    @pyqtSlot()
    def convolution_action(self):
        print("convolution_action")

    @pyqtSlot()
    def rotate_action(self):
        degrees, ok = QInputDialog.getText(self, 'Rotation', 'Select amount you would like to rotate (in degrees)')
        if ok:
            try:
                degrees = int(degrees)
                self.output_pixmap.rotate(degrees)
                self.output_label.setPixmap(self.output_pixmap)
            except ValueError:
                msg = QMessageBox()
                msg.setText(f"Error not a valid rotation amount")
                msg.exec_()

    #########
    # Utils #
    #########

    def get_input_image(self):
        pass

    def get_output_image(self):
        pass