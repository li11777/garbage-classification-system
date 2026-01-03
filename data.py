import os
import random
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0, test_scale=0.2):
    # 只获取数据集文件夹中的类别目录
    class_names = [d for d in os.listdir(src_data_folder) if os.path.isdir(os.path.join(src_data_folder, d))]

    # 在目标目录下创建train, val, test文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        os.makedirs(split_path, exist_ok=True)
        # 在各分割下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            os.makedirs(class_split_path, exist_ok=True)

    # 遍历每个类别
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        # 只处理图像文件
        current_all_data = [f for f in os.listdir(current_class_data_path) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(target_data_folder, 'train', class_name)
        val_folder = os.path.join(target_data_folder, 'val', class_name)
        test_folder = os.path.join(target_data_folder, 'test', class_name)

        train_stop = int(current_data_length * train_scale)
        val_stop = int(current_data_length * (train_scale + val_scale))

        for idx, file_index in enumerate(current_data_index_list):
            src_img_path = os.path.join(current_class_data_path, current_all_data[file_index])
            if idx < train_stop:
                copy2(src_img_path, train_folder)
            elif idx < val_stop:
                copy2(src_img_path, val_folder)
            else:
                copy2(src_img_path, test_folder)

    print("数据集划分完成。")


if __name__ == "__main__":
    src_data_folder = "laji1"
    target_data_folder = "laji2"
    data_set_split(src_data_folder, target_data_folder)
