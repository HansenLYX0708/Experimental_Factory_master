import sys

from PyQt5.QtWidgets import (QApplication, QWidget, QToolTip, QPushButton,
                             QMessageBox,
                             QDesktopWidget)

from PyQt5.QtGui import QFont, QCloseEvent
from PyQt5.QtCore import QCoreApplication

class Demo3(QWidget):
    def __init__(self):
        super().__init__()
        ## 不要写  super(Demo2, self).__init__()， 会显示不出来
        self.initGUI()

    def initGUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setToolTip('this is a <b>QWidget</b> widget')

        btn = QPushButton('button', self)
        btn.setToolTip('this is a button')
        btn.resize(btn.sizeHint())
        btn.move(50,40)

        btn2 = QPushButton('Quit', self)
        btn2.clicked.connect(QCoreApplication.instance().quit)
        btn2.setToolTip('this is a 推出 button')
        btn2.resize(btn.sizeHint())
        btn2.move(200, 40)

        self.center()
        self.show()

    def closeEvent(self,event):
        reply = QMessageBox.question(self, 'Message', 'Are you sure quit?', QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Demo3()
    sys.exit(app.exec_())