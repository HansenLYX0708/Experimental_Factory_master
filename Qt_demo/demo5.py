import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication,
                             QHBoxLayout, QVBoxLayout)
from PyQt5.QtCore import QCoreApplication

class demo5(QWidget):
    def __init__(self):
        super().__init__()
        self.initGUI()

    def initGUI(self):
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")

        hbox = QHBoxLayout()
        hbox.addStretch(0)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)

        vbox = QVBoxLayout()
        vbox.setDirection(QVBoxLayout.BottomToTop)
        vbox.addStretch(0)

        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Buttons')
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = demo5()
    sys.exit(app.exec_())