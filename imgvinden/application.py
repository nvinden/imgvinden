import sys
import os
import math
from copy import deepcopy, copy

from .dialogues import CropDialog, ConvolutionDialog, NLFDialog, HistogramDialog
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

        #self.convolution(np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype = np.int32))

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

    def crop(self, x_range : tuple[float, float], y_range : tuple[float, float]):
        image = self.image_to_np()

        x_values_range = range(int(x_range[0] * image.shape[0]), int(x_range[1] * image.shape[0]))
        y_values_range = range(int(y_range[0] * image.shape[1]), int(y_range[1] * image.shape[1]))

        out_image = np.zeros(shape = [len(list(x_values_range)), len(list(y_values_range)), 3], dtype = np.uint8)

        crop_shift_x = list(x_values_range)[0]
        crop_shift_y = list(y_values_range)[0]

        for x in x_values_range:
            for y in y_values_range:
                out_image[x - crop_shift_x, y - crop_shift_y, :] = image[x, y, :]

        self.use_np_image(out_image, save_name="crop")

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

        x_min = int(min(tl[0], bl[0], tr[0], br[0]))
        x_max = int(max(tl[0], bl[0], tr[0], br[0]))

        y_min = int(min(tl[1], bl[1], tr[1], br[1]))
        y_max = int(max(tl[1], bl[1], tr[1], br[1]))

        x_range = int(x_max - x_min) + 1
        y_range = int(y_max - y_min) + 1

        rotated = np.zeros(shape = [x_range, y_range, 3], dtype=np.uint8)
        
        x_translation = 0
        y_translation = 0

        # TODO add bilinear interpolation?? Might be fun

        for x in range(x_range):
            for y in range(y_range):
                x_prime, y_prime = rotate_point(x + x_min, y + y_min, alpha = - rotation_rad)

                if x_prime >= height or x_prime < 0:
                    rotated[x, y, :] = 0
                    continue
                if y_prime >= width or y_prime < 0:
                    rotated[x, y, :] = 0
                    continue

                x_prime = int(max(0, min(x_prime + 0.5, image.shape[0] - 1)))
                y_prime = int(max(0, min(y_prime + 0.5, image.shape[1] - 1)))

                rotated[x, y, :] = image[x_prime, y_prime, :]

        self.use_np_image(rotated, save_name="rotate")

    def scale(self, scale_factor : float):
        image = self.image_to_np()

        height, width, _ = image.shape

        out_image = np.zeros(shape = [int(height * scale_factor) + 1, int(width * scale_factor) + 1, 3], dtype = np.uint8)

        for x in range(out_image.shape[0]):
            for y in range(out_image.shape[1]):
                x_1 = int(float(x) / scale_factor)
                x_2 = x_1 + 1

                y_1 = int(float(y) / scale_factor)
                y_2 = y_1 + 1

                x_t = x / scale_factor
                y_t = y / scale_factor

                # Border cases:
                if x_2 >= image.shape[0] and y_2 >= image.shape[1]:
                    out_image[x, y] = image[image.shape[0] - 1 , image.shape[1] - 1, :]
                    continue
                elif x_2 >= image.shape[0]:
                    out_image[x, y] = image[image.shape[0] - 1 , y_1, :]
                    continue
                elif y_2 >= image.shape[1]:
                    out_image[x, y] = image[x_1 , image.shape[1] - 1, :]
                    continue

                q_11 = image[x_1, y_1, :]
                q_12 = image[x_1, y_2, :]
                q_21 = image[x_2, y_1, :]
                q_22 = image[x_2, y_2, :]

                # Actual bilinear interpolation
                r_1 = q_11 * (x_2 - x_t) + q_21 * (x_t - x_1)
                r_2 = q_12 * (x_2 - x_t) + q_22 * (x_t - x_1)
                p =   r_1  * (y_2 - y_t) + r_2  * (y_t - y_1)

                out_image[x, y, :] = p.astype(np.uint8)

        self.use_np_image(out_image, save_name = "scale")

    def PLGLM(self, gamma : float):
        image = self.image_to_np()

        def gamma_transformation(u):
            return (np.power(u / 255, gamma) * 255).astype(np.uint8)

        out_image = np.zeros_like(image, dtype = np.uint8)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                out_image[x, y, :] = gamma_transformation(image[x, y, :])
        
        self.use_np_image(out_image, save_name = "PLGLM")

    def LGLM(self, gain : float, bias : float):
        image = self.image_to_np()

        def gamma_transformation(u):
            return np.clip(gain * u + bias, 0, 255).astype(np.uint8)

        out_image = np.zeros_like(image, dtype = np.uint8)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                out_image[x, y, :] = gamma_transformation(image[x, y, :])
        
        self.use_np_image(out_image, save_name = "LGLM")

    def convolution(self, kernel : np.ndarray):
        image = self.image_to_np()

        out_image = np.zeros_like(image, dtype = np.int32)

        x_mid_conv = int((kernel.shape[0] - 1) / 2)
        y_mid_conv = int((kernel.shape[1] - 1) / 2)

        padded_image = np.pad(image, pad_width = [(x_mid_conv, ), (y_mid_conv, ), (0, )], mode='constant', constant_values=0)
        padded_image = padded_image[:, :, :3]

        kernel_t = np.flip(np.flip(kernel, 0), 1)
        kernel_t = np.stack((kernel_t, kernel_t, kernel_t), axis = 2)

        for x in range(x_mid_conv, image.shape[0] + x_mid_conv):
            for y in range(y_mid_conv, image.shape[1] + y_mid_conv):
                image_patch = padded_image[x - x_mid_conv : x + x_mid_conv + 1, y - y_mid_conv : y + y_mid_conv + 1, :]
                result = np.sum(np.multiply(kernel_t, image_patch), axis = (0, 1))
                out_image[x - x_mid_conv, y - y_mid_conv, :] = result

        # Normalzing the image
        img_max = np.max(out_image)
        img_min = np.min(out_image)
        out_image = np.asarray(255 * (out_image - img_min) / (img_max - img_min), dtype = np.uint8)

        self.use_np_image(out_image, save_name = "Convolution")

    def NLF(self, filter_type : str):
        assert filter_type in ["Minimum", "Median", "Maximum"]    

        image = self.image_to_np()

        out_image = np.zeros_like(image, dtype = np.uint8) 

        def in_picture(x, y):
            return (x >= 0 and x <= image.shape[0] - 1) and (y >= 0 and y <= image.shape[1] - 1)
            
        surround_indexes = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0)]

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                adjacent_pixels = list()

                for idx in surround_indexes:
                    if in_picture(x + idx[0], y + idx[1]): adjacent_pixels.append(image[x + idx[0], y + idx[1]])

                adjacent_pixels.sort(key = lambda x: np.mean(x))

                if filter_type == "Minimum":
                    chosen_pixel = adjacent_pixels[0]
                elif filter_type == "Median":
                    if len(adjacent_pixels) % 2 == 1:
                        chosen_pixel = adjacent_pixels[(len(adjacent_pixels) - 1) // 2]
                    else:
                        first_pixel = adjacent_pixels[(len(adjacent_pixels) - 1) // 2]
                        seond_pixel = adjacent_pixels[(len(adjacent_pixels) + 1) // 2]
                        chosen_pixel = (first_pixel + seond_pixel) / 2
                elif filter_type == "Maximum":
                    chosen_pixel = adjacent_pixels[-1]

                out_image[x, y, :] = chosen_pixel

        self.use_np_image(out_image, save_name = "Convolution")

    def histogram_equalization(self, histogram : list):
        image = self.image_to_np()

        out_image = np.zeros_like(image, dtype = np.uint8)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                new_col_values = np.array([255 * histogram[val] for val in image[x, y, :]], dtype = np.uint8)
                #new_col_values = np.array([255 * histogram[int(np.mean(image[x, y]))]] * 3, dtype=np.uint8)
                out_image[x, y, :] = new_col_values

        self.use_np_image(out_image, save_name = "hist-equalization")

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

        self.swap_image_button = QPushButton("SWAP", self)
        self.swap_image_button.clicked.connect(self.swap_image_action)
        self.swap_image_button.width = 30

        self.image_grid.setColumnStretch(0, 10)
        self.image_grid.setColumnStretch(1, 1)
        self.image_grid.setColumnStretch(2, 10)

        self.image_grid.setRowMinimumHeight(0, 750)

        self.image_grid.addWidget(self.input_label, 0, 0)
        self.image_grid.addWidget(self.swap_image_button, 0, 1)
        self.image_grid.addWidget(self.output_label, 0, 2)

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

        self.NLF_button = QPushButton("Non-Linear Filtering", self)
        self.NLF_button.clicked.connect(self.NLF_action)
        self.button_grid.addWidget(self.NLF_button, 2, 2)

        self.histogram_button = QPushButton("Histogram", self)
        self.histogram_button.clicked.connect(self.histogram_action)
        self.button_grid.addWidget(self.histogram_button, 3, 1)

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
    def swap_image_action(self):
        print("swap action")

        output_image_path = self.output_pixmap.image_path
        self.output_pixmap.original_image_path = output_image_path
        self.output_pixmap.image_path = output_image_path
        self.input_pixmap = self.output_pixmap
        self.input_label.setPixmap(self.input_pixmap)

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

        def activate_crop(values):
            try:
                x_min = float(values['X_min'])
                x_max = float(values['X_max'])
                y_min = float(values['Y_min'])
                y_max = float(values['Y_max'])

                if x_min > x_max:
                    raise Exception
                if y_min > y_max:
                    raise Exception
                
                if x_min < 0.0 or x_max > 1.0:
                    raise Exception
                if y_min < 0.0 or y_max > 1.0:
                    raise Exception

                self.output_pixmap.crop(x_range = (x_min, x_max), y_range = (y_min, y_max))
                self.output_label.setPixmap(self.output_pixmap)

            except:
                msg = QMessageBox()
                msg.setText(f"You entered an illegal crop combination")
                msg.exec_()

        dg = CropDialog()
        dg.accepted.connect(activate_crop)
        dg.exec_()

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

        gain, ok = QInputDialog.getText(self, 'Linear Gray-Level Mapping', 'What gain number would you like to use? (ex. 2.0)')
        if ok:
            try:
                gain = float(gain) 

                bias, ok = QInputDialog.getText(self, 'Linear Gray-Level Mapping', 'What bias number would you like to use? (ex. 2.0)')
                if ok:
                    try:
                        bias = float(bias)

                        self.output_pixmap.LGLM(gain, bias)
                        self.output_label.setPixmap(self.output_pixmap)
                    except ValueError:
                        msg = QMessageBox()
                        msg.setText(f"Error not a valid bias")
                        msg.exec_()
            except ValueError:
                msg = QMessageBox()
                msg.setText(f"Error not a valid gain")
                msg.exec_()

    @pyqtSlot()
    def PLGLM_action(self):
        print("PLGLM_action")

        gamma, ok = QInputDialog.getText(self, 'Power-Law Mapping', 'What gamma number would you like to use? (ex. 2.0)')
        if ok:
            try:
                gamma = float(gamma)
                self.output_pixmap.PLGLM(gamma)
                self.output_label.setPixmap(self.output_pixmap)
            except ValueError:
                msg = QMessageBox()
                msg.setText(f"Error not a valid rotation amount")
                msg.exec_()
    
    @pyqtSlot()
    def convolution_action(self):
        print("convolution_action")

        def activate_conv(values):
            #try:
            kernel = list()

            # String to lists
            rows = values['kernel'].split("\n")
            for row in rows:
                kernel.append([int(val) for val in row.split(" ")])

            # Padding out the empty (or shorten) rows
            max_row_lengths = max([len(row) for row in kernel])
            for row in kernel:
                extension = [0] * (max_row_lengths - len(row))
                row.extend(extension)

            kernel = np.asarray(kernel, dtype=np.int32)

            if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
                msg = QMessageBox()
                msg.setText(f"Error must have an odd number of rows and columns in your kernel")
                msg.exec_()
                return

            self.output_pixmap.convolution(kernel)
            self.output_label.setPixmap(self.output_pixmap)     

            #except ValueError:
            #    msg = QMessageBox()
            #    msg.setText(f"Error invalid convolutional kernel")
            #    msg.exec_()


        dg = ConvolutionDialog()
        dg.accepted.connect(activate_conv)
        dg.exec_()

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

    @pyqtSlot()
    def NLF_action(self):
        print("NLF_action")

        def activate_NLF(values):
            response = values["response"]
            self.output_pixmap.NLF(response)
            self.output_label.setPixmap(self.output_pixmap)

        dg = NLFDialog()
        dg.accepted.connect(activate_NLF)
        dg.exec_()

    @pyqtSlot()
    def histogram_action(self):

        def activate_hist(values):
            histogram = values["histogram"]
            self.output_pixmap.histogram_equalization(histogram)
            self.output_label.setPixmap(self.output_pixmap)

        dg = HistogramDialog(self.input_pixmap.image_to_np())
        dg.accepted.connect(activate_hist)
        dg.exec_()