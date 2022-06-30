import sys

from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton
from PyQt5.QtGui import QFont


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

        self.setGeometry(300, 300, 500, 400)
        self.setWindowTitle('tooltips')
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Demo3()
    sys.exit(app.exec_())