import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import PyQt5
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap, QImage

# This class is inspired by techniques I found at this website
# https://stackoverflow.com/questions/60645050/pyqt5-get-input-with-multiple-values
class ConvolutionDialog(QDialog):

    accepted = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.kernel = QPlainTextEdit()

        self.header = QLabel()
        self.header.setText("Add Convolutional Kernel (use spaces to differentiate)")

        self.btn = QPushButton('OK')
        self.btn.setDisabled(False)
        self.btn.clicked.connect(self.ok_pressed)

        form = QFormLayout(self)
        form.addRow(self.header)
        form.addRow(self.kernel)
        form.addRow(self.btn)

    def unlock(self, text):
        if text:
            self.btn.setEnabled(True)
        else:
            self.btn.setDisabled(True)

    def ok_pressed(self):
        values = {'kernel': self.kernel.toPlainText()}
        self.accepted.emit(values)
        self.accept()

class CropDialog(QDialog):
    
    accepted = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.X_min = QLineEdit()
        self.X_min.textEdited[str].connect(self.unlock)

        self.X_max = QLineEdit()
        self.X_max.textEdited[str].connect(self.unlock)

        self.Y_min = QLineEdit()
        self.Y_min.textEdited[str].connect(self.unlock)

        self.Y_max = QLineEdit()
        self.Y_max.textEdited[str].connect(self.unlock)

        self.btn = QPushButton('OK')
        self.btn.setDisabled(True)
        self.btn.clicked.connect(self.ok_pressed)

        form = QFormLayout(self)
        form.addRow('X Minimum [0, 1]', self.X_min)
        form.addRow('X Maximum [0, 1]', self.X_max)
        form.addRow('Y Minimum [0, 1]', self.Y_min)
        form.addRow('Y Maximum [0, 1]', self.Y_max)
        form.addRow(self.btn)

    def unlock(self, text):
        if text:
            self.btn.setEnabled(True)
        else:
            self.btn.setDisabled(True)

    def ok_pressed(self):
        values = {'X_min': self.X_min.text(),
                  'X_max': self.X_max.text(),
                  'Y_min': self.Y_min.text(),
                  'Y_max': self.Y_max.text()}
        self.accepted.emit(values)
        self.accept()

# Non linear filtering dialog
class NLFDialog(QDialog):
    accepted = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.title_label = QLabel("Selet type of non-linear filtering")

        # Buttons
        self.minimum = QPushButton("Minimum")
        self.minimum.clicked.connect(self.min_action)

        self.median = QPushButton("Median")
        self.median.clicked.connect(self.med_action)

        self.maximum = QPushButton("Maximum")
        self.maximum.clicked.connect(self.max_action)

        # Adding to forms
        form = QFormLayout(self)
        form.addRow(self.title_label)
        form.addRow(self.minimum)
        form.addRow(self.median)
        form.addRow(self.maximum)

    @pyqtSlot()
    def min_action(self):
        self.response = "Minimum"
        self.submit()

    def med_action(self):
        self.response = "Median"
        self.submit()

    def max_action(self):
        self.response = "Maximum"
        self.submit()

    # Returning values
    def submit(self):
        self.close()
        self.accepted.emit({"response": self.response})
        self.accept()
    
class HistogramDialog(QDialog):
    accepted = pyqtSignal(dict)

    def __init__(self, image : np.ndarray):
        super().__init__()

        self.setWindowTitle("Histograms")

        self.image = image

        self.title_label = QLabel("Image Histogram")

        # master grid
        self.master_grid = QGridLayout()

        # image grid
        self.image_grid = QGridLayout()

        self.hist_label = QLabel()
        self.hist_label.setScaledContents(False)
        self.hist_label.setAlignment(Qt.AlignCenter)
        self.hist_label.setStyleSheet("border: 3px solid black;")

        self.image_grid.addWidget(self.hist_label)

        # Button grid
        self.button_grid = QGridLayout()

        self.histogram_button = QPushButton("Histogram", self)
        self.histogram_button.clicked.connect(self.histogram_action)
        self.button_grid.addWidget(self.histogram_button, 1, 0)

        self.normalized_button = QPushButton("Normalized", self)
        self.normalized_button.clicked.connect(self.normalized_action)
        self.button_grid.addWidget(self.normalized_button, 0, 0)

        self.cumulative_button = QPushButton("Cumulative", self)
        self.cumulative_button.clicked.connect(self.cumulative_action)
        self.button_grid.addWidget(self.cumulative_button, 0, 1)

        self.cumnorm_button = QPushButton("Cumulative Normalized", self)
        self.cumnorm_button.clicked.connect(self.cumnorm_action)
        self.button_grid.addWidget(self.cumnorm_button, 0, 2)

        self.equalization_button = QPushButton("Do Histogram Equalization", self)
        self.equalization_button.clicked.connect(self.equalization_action)
        self.button_grid.addWidget(self.equalization_button, 1, 2)

        self.master_grid.addLayout(self.image_grid, 0, 0)
        self.master_grid.addLayout(self.button_grid, 1, 0)

        self.setLayout(self.master_grid)

        # Running the histogram
        self.histogram_action()

        # show all the widgets
        self.show()

    ######################
    # PLOTTING FUNCTIONS #
    ######################

    # These create matplotlib histograms

    # Returns the histogram values, so the imagevinden can preform
    # histogram equalization
    @pyqtSlot()
    def equalization_action(self):
        hist_list = [0] * 256

        vals, counts = np.unique(self.image, return_counts = True)

        for val, count in zip(vals, counts):
            hist_list[val] = count

        hist_list = np.cumsum(hist_list)
        hist_list = [num / (self.image.shape[0] * self.image.shape[1] * 3) for num in hist_list]

        values = {'histogram': hist_list}
        self.accepted.emit(values)
        self.accept()

    @pyqtSlot()
    def normalized_action(self):
        hist_list = [0] * 256

        vals, counts = np.unique(self.image, return_counts = True)

        for val, count in zip(vals, counts):
            hist_list[val] = count

        hist_list = [num / (self.image.shape[0] * self.image.shape[1] * 3) for num in hist_list]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.bar([i for i in range(256)], hist_list)
        ax.set_xlabel("Gray Level Values")
        ax.set_ylabel("Number of Occurances")

        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Setting the image

        height, width, channel = data.shape
        bytesPerLine = 3 * width

        image = QImage(data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.hist_pixmap = QPixmap.fromImage(image)
        self.hist_label.setPixmap(self.hist_pixmap)

    @pyqtSlot()
    def cumulative_action(self):
        hist_list = [0] * 256

        vals, counts = np.unique(self.image, return_counts = True)

        for val, count in zip(vals, counts):
            hist_list[val] = count

        hist_list = np.cumsum(hist_list)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.bar([i for i in range(256)], hist_list)
        ax.set_xlabel("Gray Level Values")
        ax.set_ylabel("Number of Occurances")

        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Setting the image

        height, width, channel = data.shape
        bytesPerLine = 3 * width

        image = QImage(data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.hist_pixmap = QPixmap.fromImage(image)
        self.hist_label.setPixmap(self.hist_pixmap)

    @pyqtSlot()
    def cumnorm_action(self):
        hist_list = [0] * 256

        vals, counts = np.unique(self.image, return_counts = True)

        for val, count in zip(vals, counts):
            hist_list[val] = count

        hist_list = np.cumsum(hist_list)
        hist_list = [num / (self.image.shape[0] * self.image.shape[1] * 3) for num in hist_list]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.bar([i for i in range(256)], hist_list)
        ax.set_xlabel("Gray Level Values")
        ax.set_ylabel("Number of Occurances")

        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Setting the image

        height, width, channel = data.shape
        bytesPerLine = 3 * width

        image = QImage(data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.hist_pixmap = QPixmap.fromImage(image)
        self.hist_label.setPixmap(self.hist_pixmap)

    @pyqtSlot()
    def histogram_action(self):
        hist_list = [0] * 256

        vals, counts = np.unique(self.image, return_counts = True)

        for val, count in zip(vals, counts):
            hist_list[val] = count

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.bar([i for i in range(256)], hist_list)
        ax.set_xlabel("Gray Level Values")
        ax.set_ylabel("Number of Occurances")

        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Setting the image

        height, width, channel = data.shape
        bytesPerLine = 3 * width

        image = QImage(data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.hist_pixmap = QPixmap.fromImage(image)
        self.hist_label.setPixmap(self.hist_pixmap)


        