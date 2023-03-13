import sys

from PyQt5.QtWidgets import QApplication

from imgvinden.application import ImgVindenGUI


def main():
    App = QApplication(sys.argv)

    gui = ImgVindenGUI()
    
    App.exec()

if __name__ == "__main__":
    main()