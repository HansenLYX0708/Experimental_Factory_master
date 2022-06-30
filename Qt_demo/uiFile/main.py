import sys
from PyQt5.QtWidgets import QApplication, QDialog
from untitled import Ui_Dialog
import style2_rc

class MyMainForm(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__()
        self.setupUi(self)
        self.pushButton_5.clicked.connect(self.display)
        self.pushButton_6.clicked.connect(self.close)

    def display(self):
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()
        self.textBrowser.setText("登录成功!\n" + "用户名是: " + username + ",密码是： " + password)


class CommonHelper:
 def __init__(self):
    pass
 @staticmethod
 def readQss(style):
    with open(style, 'r') as f:
        return f.read()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    styleFile = 'Ubuntu.qss'
    style = CommonHelper.readQss(styleFile)

    myWin = MyMainForm()
    myWin.setStyleSheet(style)
    myWin.show()
    sys.exit(myWin.exec_())