from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL.ImageQt import ImageQt
import sys
import os
import json
import numpy as np
import pyperclip
import fractal
from fractal import Fractal
from preference import Preference


class Main(QMainWindow):
    config = None
    now_position = np.array([0., 0.], np.float64)  # center point
    unit = 4. / 800  # gap among pixels

    isResizing = False

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        uic.loadUi("./ui/main.ui", self)
        self.actionApplyDetail.triggered.connect(self.menuApplyDetail)
        self.actionCopyDetail.triggered.connect(self.menuCopyDetail)
        self.actionExport.triggered.connect(self.menuExport)
        self.actionPreference.triggered.connect(self.menuPreference)

        self.config = json.load(open("./fractalviewer.config"))

    def draw(self):
        size = np.array([self.label.width(), self.label.height()])

        # print(self.now_position, size * self.config['resolution'],
        #       self.unit / self.config['resolution'])

        img = fractal.calcMandelbrot(self.config['fractal_type'],
                                     self.now_position,
                                     size * self.config['resolution'],
                                     self.unit / self.config['resolution'],
                                     complex(self.config['julia_c']))

        qimg = QPixmap.fromImage(ImageQt(img))
        qimg = qimg.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(qimg)

    def menuApplyDetail(self):
        detail, ok = QInputDialog.getText(self, "Enter Fractal Detail", "")
        if ok:
            pos_x, pos_y, unit = detail.split(",")
            self.now_position = np.array([float(pos_x), float(pos_y)], np.float64)
            self.unit = float(unit)
            self.draw()

    def menuCopyDetail(self):
        detail = str(self.now_position[0]) + "," + str(self.now_position[1]) + "," + str(self.unit)
        print(detail)
        pyperclip.copy(detail)

    def menuExport(self):
        fileName = QFileDialog.getSaveFileName(self, "Save Fractal Image", "", ".png;;.jpg")
        qimg = self.label.pixmap()
        qimg.save(fileName[0] + fileName[1])

    def menuPreference(self):
        w = Preference(self)
        w.show()
        w.exec()
        w.deleteLater()

    def mousePressEvent(self, event: QMouseEvent):  # set now_position
        if event.button() & Qt.LeftButton:
            size = [self.label.width(), self.label.height()]

            p = QCursor.pos()
            pos = QWidget.mapFromGlobal(self.label, p)

            self.now_position = self.now_position + np.array([pos.x() * self.unit, pos.y() * self.unit]) \
                - np.array([size[0] / 2 * self.unit, size[1] / 2 * self.unit])

            self.draw()
        elif event.button() & Qt.RightButton:
            self.now_position = np.array([0., 0.], np.float64)
            self.unit = 4. / 800
            self.draw()

        # if event.button() & Qt.RightButton:
        #     self.draw()

    def wheelEvent(self, event: QWheelEvent):  # set plane scale
        ratio = 1
        if event.angleDelta().y() > 0:
            ratio *= self.config['zoom_ratio']
        else:
            ratio /= self.config['zoom_ratio']

        # zoom just now image for fast showing
        # qimg = self.label.pixmap()
        # zoomed_qimg = qimg.scaled(qimg.size() * ratio, Qt.KeepAspectRatio)
        # self.label.setPixmap(zoomed_qimg)

        # Real Calculating for more detailed image in thread(will)
        self.unit /= ratio
        self.draw()

    def resizeEvent(self, event: QResizeEvent):
        self.draw()


def makeConfig():
    default_config = {
        'fractal_type': Fractal.MANDELBROT,
        'julia_c': "-0.70176-0.38421j",
        'zoom_ratio': 1.2,
        'resolution': 1.2,
        'colormap': "cc.m_cyclic_wrwbw_40_90_c42_s25"
    }

    json.dump(default_config, open("./fractalviewer.config", 'w'))


if __name__ == "__main__":
    if not os.path.exists("./fractalviewer.config"):
        makeConfig()

    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())
