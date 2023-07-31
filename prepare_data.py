import os
import shutil
import sys
import glob


os.makedirs("my_data/val/completion/111", exist_ok=True)
os.makedirs("my_data/val/partial/111", exist_ok=True)

gt_path = "./my_data/val/completion/111"
pt_path = "./my_data/val/partial/111"

data_path = "val_data"   # 修改为对应原数据放置位置即可
os.chdir(data_path)
completion_list = glob.glob("*f.ply")
partial_list = glob.glob("*[0-9].ply")
print(len(completion_list) == len(partial_list))
os.chdir("/home/rogerlv/Adapointr")

for l in partial_list:
    shutil.copy(os.path.join(data_path,l), pt_path)
for l in completion_list:
    shutil.copy(os.path.join(data_path,l), gt_path)