import os


def rename_tiff_files_recursive(folder_path):
    # 用于记录全局计数（全局唯一序号）
    global_counter = 1

    # 遍历主目录及其所有子目录
    for root, dirs, files in os.walk(folder_path):
        # 筛选出当前目录中的 .tiff 文件
        tiff_files = [f for f in files if f.lower().endswith('.tiff')]

        for file in tiff_files:
            # 生成新的文件名，添加序号前缀
            new_name = f"{global_counter:03d}_{file}"  # 序号为三位数字
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, new_name)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

            # 增加全局计数器
            global_counter += 1

def rename_tiff_files_with_prefix(folder_path):
    # 获取文件夹下的所有文件
    files = os.listdir(folder_path)

    # 筛选出以 .tiff 结尾的文件
    tiff_files = [f for f in files if f.lower().endswith('.tiff')]

    # 遍历 tiff 文件并重命名
    for idx, file in enumerate(tiff_files, start=1):
        # 生成新的文件名
        new_name = f"{idx:03d}_{file}"  # 添加三位数字前缀
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")


# 指定文件夹路径
folder_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\"  # 替换为你的文件夹路径

# 调用函数
rename_tiff_files_recursive(folder_path)
