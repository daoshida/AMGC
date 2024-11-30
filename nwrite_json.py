from pathlib import Path
import random
import os


main_path = Path.cwd()
print(f'main_path:{main_path}')
data_path = Path.cwd()/'test_9c2'
data_file_list = []
label_list = []
folder_list = []

l = list(data_path.iterdir())

for i in l:
    os.chdir(i)
    temp_file_list = list(i.glob("**/*.npy"))
    temp_label_list = [int(i.name)] * len(temp_file_list)
    data_file_list += temp_file_list
    label_list += temp_label_list
    folder_list.append(str(i.name))

os.chdir(main_path)


# 所有数据一起打乱
all_data = list(zip(label_list, data_file_list))
# random.seed(100)
random.shuffle(all_data)
label_data = [i[0] for i in all_data]
data = [i[1] for i in all_data]


# 写入json文件
fo = open('Net_test1' + ".json", "w")

fo.write('{"label_names": [')
fo.writelines(['"%s",' % item for item in folder_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"data_path": [')
fo.writelines(['"%s",' % item for item in data])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"data_labels": [')
fo.writelines(['%d,' % item for item in label_data])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write(']}')

fo.close()

print("Save train.json done")