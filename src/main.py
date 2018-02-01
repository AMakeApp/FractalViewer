from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL.ImageQt import ImageQt
import sys
import os.path
import time
import threading
import json
import numpy as np
import fractal
from fractal import Fractal
from preference import Preference


class Main(QMainWindow):
    config = None
    now_position = np.array([0., 0.])  # center point
    unit = 4. / 800  # gap among pixels

    isResizing = False

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        uic.loadUi("./ui/main.ui", self)
        self.actionPreference.triggered.connect(self.configPreference)
        # self.installEventFilter(self)

        self.config = json.load(open("./fractalviewer.config"))

        # self.draw()

    def draw(self):
        img = None
        size = np.array([self.label.width(), self.label.height()])

        if self.config['fractal_type'] == Fractal.MANDELBROT:
            img = fractal.drawmandelbrot(self.now_position, size * self.config['resolution'],
                                         self.unit / self.config['resolution'])
        elif self.config['fractal_type'] == Fractal.JULIA:
            img = fractal.drawjulia(self.now_position, size * self.config['resolution'],
                                    self.unit / self.config['resolution'],
                                    complex(self.config['julia_c']))

        qimg = QPixmap.fromImage(ImageQt(img))
        qimg = qimg.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(qimg)

    def configPreference(self):
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

    def eventFilter(self, obj: QObject, event: QEvent):
        # For check resizeEvent endpoint
        if self.isResizing:
            # print(event.type())
            if event.type() == QEvent.MouseButtonRelease or event.type() == QEvent.NonClientAreaMouseButtonRelease:

                if QMouseEvent(event).button() == Qt.LeftButton:
                    self.draw()
                    self.isResizing = False
                    print("resizeEvent done!")

        return QObject.eventFilter(self, obj, event)

    def resizeEvent(self, event: QResizeEvent):
        self.isResizing = True
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
