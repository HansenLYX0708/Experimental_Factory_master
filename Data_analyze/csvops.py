import pandas as pd
import os

def mergeCsv(folderPath, savePath, saveName):
    fileList = os.listdir(folderPath)
    df = pd.read_csv(folderPath + "\\" + fileList[0])

    for i in range(1, len(fileList)):
        df1 = pd.read_csv(folderPath + "\\" + fileList[i])
        name = "item" + str(i)
        df[name] = df1["value"]
    df.to_csv(savePath + "\\" + saveName, encoding="utf_8_sig", index=False, header=False)


if __name__ == '__main__':
    folder_path = "C:\\__work\\slider_inspection\\master\\visualdl-scalar-loss"
    save_name = "merge.csv"
    mergeCsv(folder_path, folder_path, save_name)
