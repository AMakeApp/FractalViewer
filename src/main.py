import sys
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL.ImageQt import ImageQt
import time
import threading
import numpy as np
import fractal

form_class = uic.loadUiType("main.ui")[0]


class Form(QMainWindow, form_class):
    MANDELBROT = 1
    JULIA = 2

    global sort
    now_position = np.array([0., 0.])  # center point
    unit = 4. / 800  # gap among pixels
    scale = 1.2
    zoom_ratio = 1.2

    isResizing = False

    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.installEventFilter(self)
        # self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)
        # self.setFixedSize(self.size())

        self.sort = self.MANDELBROT
        self.draw()

    def draw(self):
        global img

        size = np.array([self.label.width(), self.label.height()])

        if self.sort is self.MANDELBROT:
            img = fractal.drawmandelbrot(self.now_position, size * self.scale,
                                         self.unit / self.scale)
        elif self.sort is self.JULIA:
            img = fractal.drawjulia(self.now_position, size * self.scale,
                                    self.planesize, -0.8 + 0.156j)

        qimg = QPixmap.fromImage(ImageQt(img))
        qimg = qimg.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(qimg)

    def mousePressEvent(self, event: QMouseEvent):  # set now_position
        if event.button() & Qt.LeftButton:
            size = [self.label.width(), self.label.height()]

            p = QCursor.pos()
            pos = QWidget.mapFromGlobal(self.label, p)

            self.now_position = self.now_position + np.array([pos.x() * self.unit, pos.y() * self.unit]) \
                - np.array([size[0] / 2 * self.unit, size[1] / 2 * self.unit])

            self.draw()

        if event.button() & Qt.RightButton:
            self.draw()

    def wheelEvent(self, event: QWheelEvent):  # set plane scale
        global ratio
        ratio = 1
        if event.angleDelta().y() > 0:
            ratio *= self.zoom_ratio
        else:
            ratio /= self.zoom_ratio

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
            if event.type() is QEvent.MouseButtonRelease or event.type() is QEvent.NonClientAreaMouseButtonRelease:

                if QMouseEvent(event).button() is Qt.LeftButton:
                    self.draw()
                    self.isResizing = False
                    print("resizeEvent done!")

        return QObject.eventFilter(self, obj, event)

    def resizeEvent(self, event: QResizeEvent):
        self.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Form()
    w.show()
    sys.exit(app.exec())
