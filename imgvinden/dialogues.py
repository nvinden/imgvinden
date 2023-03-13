import sys
import os

import PyQt5
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap

class InputImagetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

    def getValue(self):
        # return the current value of the spinbox
        return self.spinBox.value()
