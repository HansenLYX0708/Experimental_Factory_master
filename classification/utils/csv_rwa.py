import csv

'''
w：以写方式打开， 
a：以追加模式打开 (从 EOF 开始, 必要时创建新文件) 
r+：以读写模式打开 
w+：以读写模式打开 (参见 w ) 
a+：以读写模式打开 (参见 a ) 
rb：以二进制读模式打开 
wb：以二进制写模式打开 (参见 w ) 
ab：以二进制追加模式打开 (参见 a ) 
rb+：以二进制读写模式打开 (参见 r+ ) 
wb+：以二进制读写模式打开 (参见 w+ ) 
ab+：以二进制读写模式打开 (参见 a+ )
'''

def create_csv():
    path = "aa.csv"
    with open(path,'wb') as f:
        csv_write = csv.writer(f)
        csv_head = ["good","bad"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    with open(path,'a+', newline='') as f:
        csv_write = csv.writer(f)
        #data_row = ["1","2"]
        csv_write.writerow(data_row)

def read_csv():
    path = "aa.csv"
    with open(path,"rb") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            print(line)