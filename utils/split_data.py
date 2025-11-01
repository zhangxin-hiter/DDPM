"""
split_data.py

分割数据集，将指定数据集分割为训练集和测试集，可指定分割比
"""

import random
import os
from shutil import copy

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

def main():

    # 数据集地址
    file_path = "assets/flower_photos"
    # 数据集包含种类
    data_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla and "train" not in cla and "val" not in cla]

    file_train_path = os.path.join(file_path, "train")
    mkfile(file_train_path)
    for cla in data_class:
        mkfile(os.path.join(file_train_path, cla))

    file_val_path = os.path.join(file_path, "val")
    mkfile(file_val_path)
    for cla in data_class: 
        mkfile(os.path.join(file_val_path, cla))
    
    # 测试集分割比，可自定义
    split_rate = 0.1
    for cla in data_class:
        cla_path = os.path.join(file_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num * split_rate))
        for index, img in enumerate(images):
            if img in eval_index:
                img_path = os.path.join(cla_path, img)
                new_path = os.path.join(file_val_path, cla, img)
                copy(img_path, new_path)
            else:
                img_path = os.path.join(cla_path, img)
                new_path = os.path.join(file_train_path, cla, img)
                copy(img_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
        print()

    print("processing done!")
    
if __name__ == "__main__":
    main()