import os
import shutil
import sys
import glob


os.makedirs("my_data/train/completion", exist_ok=True)
os.makedirs("my_data/train/partial", exist_ok=True)

gt_path = "./my_data/train/completion"
pt_path = "./my_data/train/partial"

data_path = "teeth_data"
os.chdir(data_path)
completion_list = glob.glob("*f.ply")
partial_list = glob.glob("*[0-9].ply")
print(len(completion_list) == len(partial_list))
os.chdir("/home/rogerlv/PoinTr")

for l in partial_list:
    shutil.copy(os.path.join(data_path,l), pt_path)
for l in completion_list:
    shutil.copy(os.path.join(data_path,l), gt_path)