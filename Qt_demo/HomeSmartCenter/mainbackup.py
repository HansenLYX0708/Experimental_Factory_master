# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.py'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
from uiFile.GUIMain import Ui_MainWindow
import uiFile.default_rc

class MyWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet()
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
