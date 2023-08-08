import os
import glob

# 修改为对应绝对路径
os.chdir("/home/rogerlv/my_anchorformer/my_data/train/completion/111")
completion_list = glob.glob("*f.ply")

for ll in completion_list:
    os.rename(ll, ll.split(".")[0][:-2]+"."+ll.split(".")[1])

file_list = glob.glob("*")

def save_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        for ll in lst:
            file.write("\""+ll.split(".")[0]+"\""+","+"\n")
        


save_list_to_txt(file_list, "/home/rogerlv/my_anchorformer/output.txt")
