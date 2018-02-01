from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL.ImageQt import ImageQt
import sys
import json
from fractal import Fractal


class Preference(QDialog):
    backup_config = None
    edited_config = None

    isEdited = False

    parent = None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        self.backup_config = parent.config
        self.edited_config = parent.config

        uic.loadUi("./ui/preference.ui", self)
        self.buttonOK.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)

        self.comboFractalType.setCurrentIndex(self.backup_config['fractal_type'])
        self.comboFractalType.currentIndexChanged.connect(self.comboIndexChanged)

        c = complex(self.backup_config['julia_c'])
        if c.imag < 0:
            c = complex(c.real, -c.imag)
            formated_c = '{0.real} - {0.imag}i'.format(c)
        elif c.imag == 0:
            formated_c = '{0}'.format(c)
        else:
            formated_c = '{0.real} + {0.imag}i'.format(c)

        self.editJuliaC.setText(formated_c)

        if self.backup_config['fractal_type'] != Fractal.JULIA:
            self.editJuliaC.setReadOnly(True)

    def comboIndexChanged(self, index):
        if index == Fractal.JULIA:
            self.editJuliaC.setReadOnly(False)
        else:
            self.editJuliaC.setReadOnly(True)

        self.edited_config['fractal_type'] = index

    def accept(self):
        json.dump(self.edited_config, open("./fractalviewer.config", 'w'))
        self.apply()

        super(Preference, self).accept()

    def reject(self):
        if self.isEdited:
            self.parent.config = self.backup_config

        super(Preference, self).reject()

    def apply(self):
        c = self.editJuliaC.text()
        c = c.replace("i", "j")
        c = c.replace(" ", "")

        # check for if it is valid complex number
        try:
            complex(c)
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("The format of complex number is not valid. Please check and edit.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()

            return

        self.isEdited = True
        self.edited_config['julia_c'] = c
        self.parent.config = self.edited_config
        self.parent.draw()  # not working now
        self.update()
