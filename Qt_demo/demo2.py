import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon

class Demo2(QWidget):
    def __init__(self):
        super().__init__()
        ## 不要写  super(Demo2, self).__init__()， 会显示不出来
        self.initGUI()

    def initGUI(self):
        self.setGeometry(300, 300, 500, 400)
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('logo.png'))
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Demo2()
    sys.exit(app.exec_())