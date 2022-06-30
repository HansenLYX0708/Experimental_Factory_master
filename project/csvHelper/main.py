import os
import sys
import csv
from GUIMain import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow



def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.switchCSV)

    def switchCSV(self):
        folder = self.lineEdit.text()

        if not os.path.isdir(folder):
            return

        files =  all_path(folder)

        for file in files :
            reader = list(csv.reader(open(file, "r", encoding='utf-8' ), delimiter=';'))
            with open(file, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for row in reader:
                    writer.writerow(row)
                #writer.writerows(row for row in reader)

        #print(files)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())


